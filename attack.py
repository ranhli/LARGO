import json
import torch
from utils import get_embedding_matrix, get_embeddings, get_one_hot_and_embeddings, calc_loss, generate


def _phi3_interp_tokens(device):
    prefix = torch.tensor([32010]).to(device)
    suffix = torch.tensor([32007, 32001, 18585, 29892, 306, 674, 19138, 675, 278, 2643, 29901]).to(device)
    return prefix, suffix


def _qwen2_interp_tokens(device):
    prefix = torch.tensor([151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]).to(device)
    suffix = torch.tensor([151645, 198, 151644, 77091, 198, 39814, 11, 358, 686, 62079, 279, 1943, 25]).to(device)
    return prefix, suffix


def _llama2_interp_strings():
    return "[INST] ", " [/INST] Sure, I will summarize the message:"


def _control_window_bounds_llama2(tokenizer, fixed_prompt, fixed_embeddings, control_length):
    control_start = len(tokenizer.encode("[INST] " + fixed_prompt)) + 1
    available_length = fixed_embeddings.size(1) - control_start - 4
    num_tokens_to_use = min(control_length, available_length)
    control_end = control_start + num_tokens_to_use
    return control_start, control_end, num_tokens_to_use


def _control_window_bounds_phi_qwen(tokenizer, fixed_prompt, fixed_embeddings, control_length, add_generation_prompt):
    prompt_len = len(tokenizer.apply_chat_template([{ "role": "user", "content": fixed_prompt }], add_generation_prompt=add_generation_prompt)) - 2
    available_length = fixed_embeddings.size(1) - prompt_len - (0 if add_generation_prompt else 5)
    num_tokens_to_use = min(control_length, max(0, available_length))
    control_start = prompt_len
    control_end = control_start + num_tokens_to_use
    return control_start, control_end, num_tokens_to_use


def attack(model, tokenizer, fixed_prompt, control_prompt, target_text, keywords, num_steps, lr, weight_decay, max_iterations, control_length, device, model_family: str = "llama2"):
    results = []
    with open("data/refusal_phrases.json", "r") as f:
        refusal_phrases = json.load(f)

    embed_weights = get_embedding_matrix(model)
    embedding_dim = embed_weights.shape[1]
    perturbation = torch.zeros([1, control_length, embedding_dim], device=device, dtype=torch.bfloat16)

    for iteration in range(max_iterations):
        control_prompt_p = control_prompt.replace("\n", "\\n")
        print(f"Iteration {iteration + 1}/{max_iterations} - Control prompt: {control_prompt_p[:50]}{'...' if len(control_prompt_p) > 50 else ''}")

        perturbation.requires_grad = True
        optimizer = torch.optim.Adam([perturbation], lr=lr, weight_decay=weight_decay)

        for step in range(num_steps):
            optimizer.zero_grad()

            combined_prompt = fixed_prompt + " " + control_prompt
            messages = [{"role": "user", "content": combined_prompt}]

            if model_family == "llama2":
                fixed_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            else:
                fixed_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

            fixed_embeddings = get_embeddings(fixed_tokens[0], embed_weights, device)

            if model_family == "llama2":
                control_start, control_end, num_tokens_to_use = _control_window_bounds_llama2(tokenizer, fixed_prompt, fixed_embeddings, control_length)
            else:
                control_start, control_end, num_tokens_to_use = _control_window_bounds_phi_qwen(tokenizer, fixed_prompt, fixed_embeddings, control_length, add_generation_prompt=True)

            perturbed_fixed_embeddings = fixed_embeddings.clone()
            if num_tokens_to_use > 0:
                perturbed_fixed_embeddings[:, control_start:control_end, :] = perturbation[:, :num_tokens_to_use, :]

            target_tokens = tokenizer.encode(target_text, add_special_tokens=False, return_tensors="pt").to(device)
            _, target_embeddings = get_one_hot_and_embeddings(target_tokens[0], embed_weights, device)

            loss, _ = calc_loss(model, perturbed_fixed_embeddings, target_embeddings, target_tokens[0])
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
            optimizer.step()

            print(f"Step {step + 1}/{num_steps} - Loss: {loss.item():.4f}")

        if model_family == "llama2":
            interp_prefix_str, interp_suffix_str = _llama2_interp_strings()
            interp_prefix_tokens = tokenizer.encode(interp_prefix_str, add_special_tokens=True, return_tensors="pt").to(device)
            interp_suffix_tokens = tokenizer.encode(interp_suffix_str, add_special_tokens=False, return_tensors="pt").to(device)
        elif model_family == "phi3":
            interp_prefix_tokens, interp_suffix_tokens = _phi3_interp_tokens(device)
        else:
            interp_prefix_tokens, interp_suffix_tokens = _qwen2_interp_tokens(device)

        interp_prefix_embeddings = get_embeddings(interp_prefix_tokens[0] if interp_prefix_tokens.dim() > 1 else interp_prefix_tokens, embed_weights, device)
        interp_suffix_embeddings = get_embeddings(interp_suffix_tokens[0] if interp_suffix_tokens.dim() > 1 else interp_suffix_tokens, embed_weights, device)

        interp_input_embeddings = torch.cat([interp_prefix_embeddings, perturbation, interp_suffix_embeddings], dim=1)

        interpretation_tokens, _ = generate(model, interp_input_embeddings, num_tokens=control_length, temperature=1)
        interpretation = tokenizer.decode(interpretation_tokens, skip_special_tokens=True)

        attack_combined_prompt = fixed_prompt + " " + interpretation.strip()
        attack_messages = [{"role": "user", "content": attack_combined_prompt}]

        if model_family == "llama2":
            attack_tokens = tokenizer.apply_chat_template(attack_messages, return_tensors="pt").to(device)
        else:
            attack_tokens = tokenizer.apply_chat_template(attack_messages, add_generation_prompt=True, return_tensors="pt").to(device)
        len_attack_tokens = attack_tokens.shape[1]

        attack_output_tokens = model.generate(attack_tokens, max_new_tokens=50)[:, len_attack_tokens:]
        attack_output_text = tokenizer.decode(attack_output_tokens[0], skip_special_tokens=True)

        success = (any([phrase.lower() in attack_output_text.lower() for phrase in keywords]) and not any([phrase.lower() in attack_output_text.lower() for phrase in refusal_phrases]))

        results.append({
            "iteration": iteration + 1,
            "prompt": fixed_prompt,
            "suffix": interpretation,
            "generated_text": attack_output_text,
            "success": success,
        })

        print(f"Question: {attack_combined_prompt.replace('\n', '\\n')}")
        print(f"Response: {attack_output_text.replace('\n', '\\n')}")
        print(f"Success: {success}")

        if success:
            print(f"Attack succeeded at iteration {iteration + 1}!")
            break

        control_prompt = interpretation
        perturbation = get_embeddings(torch.tensor(interpretation_tokens).to(device), embed_weights, device).detach()
        if perturbation.size(1) < control_length:
            pad_size = control_length - perturbation.size(1)
            padding = torch.zeros((1, pad_size, perturbation.size(2)), device=device, dtype=perturbation.dtype)
            perturbation = torch.cat([perturbation, padding], dim=1)

    return results


