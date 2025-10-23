import json
import torch
import random
from utils import get_embedding_matrix, get_embeddings, get_one_hot_and_embeddings, calc_loss, generate


def train_universal_suffix(model, tokenizer, train_examples, test_examples, suffix, num_epochs, num_steps, batch_size, lr, weight_decay, control_length, device):
    results = []
    with open("data/refusal_phrases.json", "r") as f:
        refusal_phrases = json.load(f)

    embed_weights = get_embedding_matrix(model)
    embedding_dim = embed_weights.shape[1]
    perturbation = torch.zeros([1, control_length, embedding_dim], device=device, dtype=torch.bfloat16)

    train_questions, train_answers = train_examples["goal"], train_examples["target"]
    test_questions, test_answers = test_examples["goal"], test_examples["target"]

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam([perturbation], lr=lr, weight_decay=weight_decay)
        perturbation.requires_grad = True

        data = list(zip(train_questions, train_answers))
        random.shuffle(data)
        questions_epoch, answers_epoch = zip(*data)

        batch_count = 0
        epoch_loss = 0.0
        optimizer.zero_grad()

        for _ in range(num_steps):
            for question, answer in zip(questions_epoch, answers_epoch):
                combined_prompt = question + " " + suffix
                messages = [{"role": "user", "content": combined_prompt}]
                fixed_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
                fixed_embeddings = get_embeddings(fixed_tokens[0], embed_weights, device)

                control_start = len(tokenizer.encode("[INST] " + question)) + 1
                available_length = fixed_embeddings.size(1) - control_start - 4
                num_tokens_to_use = min(control_length, available_length)
                control_end = control_start + num_tokens_to_use

                perturbed_fixed_embeddings = fixed_embeddings.clone()
                perturbed_fixed_embeddings[:, control_start:control_end, :] = perturbation[:, :num_tokens_to_use, :]

                answer_tokens = tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(device)
                _, target_embeddings = get_one_hot_and_embeddings(answer_tokens[0], embed_weights, device)

                loss, _ = calc_loss(model, perturbed_fixed_embeddings, target_embeddings, answer_tokens[0])
                loss = loss / batch_size
                loss.backward(retain_graph=True)

                batch_count += 1
                epoch_loss += loss.item()

                if batch_count % batch_size == 0 or batch_count == len(questions_epoch):
                    torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
                    optimizer.step()
                    print(f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_count // batch_size} - Loss: {loss.item() * batch_size:.4f}")

        interp_prefix = "[INST] "
        interp_suffix = " [/INST] Sure, I will summarize the message:"

        interp_prefix_tokens = tokenizer.encode(interp_prefix, add_special_tokens=True, return_tensors="pt").to(device)
        interp_suffix_tokens = tokenizer.encode(interp_suffix, add_special_tokens=False, return_tensors="pt").to(device)

        interp_prefix_embeddings = get_embeddings(interp_prefix_tokens[0], embed_weights, device)
        interp_suffix_embeddings = get_embeddings(interp_suffix_tokens[0], embed_weights, device)

        interp_input_embeddings = torch.cat([interp_prefix_embeddings, perturbation, interp_suffix_embeddings], dim=1)

        interp_generated_tokens, _ = generate(model, interp_input_embeddings, num_tokens=control_length, temperature=1)
        interpretation = tokenizer.decode(interp_generated_tokens, skip_special_tokens=True)

        test_results = []
        test_correct = 0
        for question, answer in zip(test_questions, test_answers):
            messages = [{"role": "user", "content": question + " " + interpretation.strip()}]
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            inputs_length = inputs.shape[1]
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=25)[:, inputs_length:]
            output_text = tokenizer.batch_decode(outputs)[0]

            interp_success = not any([phrase.lower() in output_text.lower() for phrase in refusal_phrases])
            if interp_success:
                test_correct += 1
                test_results.append({"question": question, "answer": output_text, "success": True})
            else:
                test_results.append({"question": question, "answer": output_text, "success": False})

        test_asr = test_correct / len(test_questions)

        epoch_result = {
            "epoch": epoch + 1,
            "suffix": interpretation,
            "loss": epoch_loss,
            "asr": test_asr,
            "results": test_results,
        }
        results.append(epoch_result)

        with open(f"data/13b_{num_epochs}_{num_steps}_{batch_size}_{lr}.json", "w") as f:
            json.dump(results, f)

        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Test ASR: {test_asr:.2f}")

        new_perturbation = get_embeddings(torch.tensor(interp_generated_tokens).to(device), embed_weights, device).detach()
        if new_perturbation.size(1) < control_length:
            pad_size = control_length - new_perturbation.size(1)
            padding = torch.zeros((1, pad_size, new_perturbation.size(2)), device=device, dtype=new_perturbation.dtype)
            new_perturbation = torch.cat([new_perturbation, padding], dim=1)
        perturbation.data.copy_(new_perturbation)

    return results


