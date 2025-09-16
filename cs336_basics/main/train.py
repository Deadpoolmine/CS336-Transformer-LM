import os

import cs336_basics.optimizer.adamw as optimizer
import cs336_basics.data.loader as loader
import cs336_basics.transformer.transformer as transformer
import cs336_basics.tokenizer.bpe_tokenizer as tokenizer
import cs336_basics.tokenizer.bpe_train as bpe_train
import cs336_basics.transformer.attn as attn
import numpy as np
import torch


train_path = os.path.join(
    "/home/deadpool/Playground/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
)
model_path = os.path.join(
    "/home/deadpool/Playground/CS336/assignment1-basics/ckpt/my_model"
)


def init_kaiming_weights(dimsions):
    weight = torch.empty(dimsions)
    torch.nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
    return weight


def init_weights(n_layers, d_model, d_ff, n_heads, batch_size):
    q_proj_weight = init_kaiming_weights((d_model, d_model))
    k_proj_weight = init_kaiming_weights((d_model, d_model))
    v_proj_weight = init_kaiming_weights((d_model, d_model))
    o_proj_weight = init_kaiming_weights((d_model, d_model))
    w1_weight = init_kaiming_weights((d_ff, d_model))
    w2_weight = init_kaiming_weights((d_model, d_ff))
    w3_weight = init_kaiming_weights((d_ff, d_model))

    # we will initialize weights randomly
    weights = {}
    weights["token_embeddings.weight"] = None
    weights["lm_head.weight"] = None
    for i in range(n_layers):
        weights.update(
            {
                f"layers.{i}.attn.q_proj.weight": torch.nn.Parameter(
                    q_proj_weight.clone()
                ),
                f"layers.{i}.attn.k_proj.weight": torch.nn.Parameter(
                    k_proj_weight.clone()
                ),
                f"layers.{i}.attn.v_proj.weight": torch.nn.Parameter(
                    v_proj_weight.clone()
                ),
                f"layers.{i}.attn.output_proj.weight": torch.nn.Parameter(
                    o_proj_weight.clone()
                ),
                f"layers.{i}.ln1.weight": None,
                f"layers.{i}.ffn.w1.weight": torch.nn.Parameter(w1_weight.clone()),
                f"layers.{i}.ffn.w2.weight": torch.nn.Parameter(w2_weight.clone()),
                f"layers.{i}.ffn.w3.weight": torch.nn.Parameter(w3_weight.clone()),
                f"layers.{i}.ln2.weight": None,
            }
        )
    weights["ln_final.weight"] = None
    return weights


def train_or_infer(
    context_length=64,
    batch_size=32,
    total_iters=10000,
    lr=1e-3,
    weight_decay=0.01,
    device="cpu",
    checkpoint_path=None,
    checkpoint_interval=1000,
    vocab_size=1000,
    d_model=512 // 8,
    d_ff=1344 // 8,
    rope_theta=10000,
    n_layers=3,
    n_heads=2,
    mode="train",
):
    device = torch.device(device)

    ref_vocab, ref_merges = bpe_train.train_bpe(
        input_path=train_path, vocab_size=500, special_tokens=["<|endoftext|>"]
    )

    t = tokenizer.Tokenizer(ref_vocab, ref_merges, ["<|endoftext|>"])
    assert (
        len(t.vocab) == vocab_size
    ), f"Vocab size mismatch: {len(t.vocab)} vs {vocab_size}"

    weights = init_weights(n_layers, d_model, d_ff, n_heads, vocab_size)
    # start from an empty weights
    m = transformer.Transformer(
        d_model=d_model,
        num_heads=n_heads,
        ffn_d_ff=d_ff,
        max_seq_len=context_length,
        num_layers=n_layers,
        vocab_size=vocab_size,
        theta=rope_theta,
        weights=weights,
    )
    # optimizer
    o = optimizer.AdamW(
        m.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if mode == "train":
        with open(train_path, "r") as f:
            data = f.read()
            tokens = t.encode(data)

            for i in range(total_iters):
                # update optimizers
                o.zero_grad()

                x, y = loader.data_loading(
                    x=tokens,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )
                logits = m(x)
                loss = attn.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                print(f"Iter {i}: loss {loss.item()}")

                loss.backward()
                optimizer.gradient_clipping(m.parameters(), max_norm=1e-2)
                o.step()

                if checkpoint_path is not None and (i + 1) % checkpoint_interval == 0:
                    torch.save(m.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        if checkpoint_path is not None:
            torch.save(m.state_dict(), checkpoint_path)
            print(f"Saved final model to {checkpoint_path}")

    elif mode == "infer":
        parameters = torch.load(model_path)
        m.load_state_dict(parameters)

        prompt = "This is a test"
        prompt_tokens = t.encode(prompt)
        max_ctx_len = 50  # number of tokens to generate
        print(f"Prompt: {prompt}")
        while True and max_ctx_len > 0:
            x = torch.tensor([prompt_tokens], device=device)

            logits = m(x)

            prob = attn.softmax(logits, dim=-1)
            next_token = torch.argmax(prob[:, -1, :], dim=-1).item()
            if next_token == 0:
                print("End of text token generated. Stopping.")
                return
            prompt_tokens.append(next_token)
            generated = t.decode(prompt_tokens)
            max_ctx_len -= 1

        print(f"Generated: {generated}")


if __name__ == "__main__":
    # train_or_infer(
    #     total_iters=1000,
    #     batch_size=32,
    #     d_model=512,
    #     d_ff=1344,
    #     context_length=256,
    #     vocab_size=500,
    #     n_layers=4,
    #     n_heads=16,
    #     device="cpu",
    #     lr=0.01,
    #     checkpoint_path="/home/deadpool/Playground/CS336/assignment1-basics/ckpt/my_model",
    #     mode="train",
    # )  # for quick test

    train_or_infer(
        total_iters=1000,
        batch_size=32,
        d_model=512,
        d_ff=1344,
        context_length=256,
        vocab_size=500,
        n_layers=4,
        n_heads=16,
        device="cpu",
        lr=0.01,
        checkpoint_path="/home/deadpool/Playground/CS336/assignment1-basics/ckpt/my_model",
        mode="infer",
    )  # for quick test
