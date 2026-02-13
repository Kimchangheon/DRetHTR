
import torch
import torch.nn.functional as F

def greedy_decode_recurrent(model, src,img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    src = src.to(device) # (2,1,384,384)
    img_patches = model.feature_extractor(src) # (2,577,384)

    # Initialize ys with batch_size x 1, filled with start_symbol
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device) #(2,1)

    s_n_1s = [
            torch.zeros(batch_size,
                        model.heads,
                        model.d_model // model.heads,
                        model.d_model // model.heads).to(device) # (2,4,64,64)
        for _ in range(model.depth) #12
    ]
    k_img_list, v_img_list = model.get_KV_img(img_patches, img_pad_ratios)
    # print("layers : ", len(x_0_to_n_1_s) )
    for i in range(max_len - 1): # 0~93
        input_word = ys[:, -1].unsqueeze(1) #(2,1)
        # input_word_embedded = model.embed_tokens(input_word) #(2,1,512)
        # b, n, _ = input_word_embedded.shape
        y_n, s_ns = model.decode_recurrent(input_word, s_n_1s, k_img_list, v_img_list, i, img_pad_ratios)
        s_n_1s = s_ns
        # out = model.decode(ys, memory, pad_idx)
        logit = model.generator(y_n) #(2,1,79)
        # print(logit.shape)
        _, next_words = torch.max(logit, dim=2) #(2,1)

        # Check for EOS_IDX for each sequence in the batch
        eos_mask = next_words == end_symbol #(2,1)

        # Update ys with the next words
        ys = torch.cat([ys, next_words], dim=1) #(2,2) --> (2,3) --> .... --> (2,max_length)
        # If all sequences in the batch have reached EOS_IDX, break the loop
        if eos_mask.all():
            break
    # Replace the elements after EOS_IDX with pad_idx for each sequence
    for idx in range(batch_size):
        eos_idx = (ys[idx] == end_symbol).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            ys[idx, eos_idx[0] + 1:] = pad_idx

    return ys

def beam_decode_recurrent(model, src, img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, beam_width, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    beam_size = beam_width
    src = src.to(device)
    batch_size = src.size(0)
    img_patches = model.feature_extractor(src)  # (batch_size, seq_len, model_dim)

    # Initialize sequences with the start symbol
    ys = torch.ones(batch_size, 1, 1).fill_(start_symbol).long().to(device)  # (batch_size, beam_size=1, seq_len=1)
    scores = torch.zeros(batch_size, 1).to(device)  # (batch_size, beam_size=1)

    # Initialize recurrent states for each layer
    s_n_1s = [
        torch.zeros(batch_size, 1, model.heads, model.d_model // model.heads, model.d_model // model.heads).to(device)
        for _ in range(model.depth)
    ]  # Each element: (batch_size, beam_size=1, heads, head_dim, v_dim_per_head)

    # Obtain key and value tensors from the image patches
    k_img_list, v_img_list = model.get_KV_img(img_patches, img_pad_ratios)
    # First decoding step
    input_word = ys[:, 0, -1].unsqueeze(1)  # (batch_size, 1)
    s_n_1s_flat = [
        s.view(batch_size * 1, model.heads, model.d_model // model.heads, model.d_model // model.heads)
        for s in s_n_1s
    ]
    y_n, s_ns = model.decode_recurrent(
        input_word, s_n_1s_flat, k_img_list, v_img_list, 0, img_pad_ratios
    )
    logits = model.generator(y_n)  # (batch_size, 1, vocab_size)
    log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size, vocab_size)

    # Select top beam_size tokens
    next_scores, next_tokens = log_probs.topk(beam_size, dim=1)  # (batch_size, beam_size)
    ys = next_tokens.unsqueeze(2)  # (batch_size, beam_size, seq_len=1)
    scores = next_scores  # (batch_size, beam_size)

    # Expand recurrent states to match beam_size
    def expand_states(s_list):
        expanded_list = []
        for s in s_list:
            s = s.unsqueeze(1)  # Shape: (batch_size, 1, heads, head_dim, v_dim_per_head)
            s = s.repeat(1, beam_size, 1, 1, 1, 1)  # (batch_size, beam_size, heads, head_dim, v_dim_per_head)
            s = s.view(
                batch_size * beam_size,
                model.heads,
                model.d_model // model.heads,
                model.d_model // model.heads,
            )
            expanded_list.append(s)
        return expanded_list

    s_n_1s = expand_states(s_ns)

    # Expand k_prev_list and v_prev_list to match beam_size
    def expand_kv_list(kv_list):
        expanded_list = []
        for kv in kv_list:
            kv = kv.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)  # (batch_size, beam_size, heads, seq_len, head_dim)
            kv = kv.view(batch_size * beam_size, *kv.shape[2:])
            expanded_list.append(kv)
        return expanded_list

    k_img_list = expand_kv_list(k_img_list)
    v_img_list = expand_kv_list(v_img_list)
    # Expand img_pad_ratios to match beam_size
    if img_pad_ratios is not None:
        img_pad_ratios = img_pad_ratios.unsqueeze(1).repeat(1, beam_size, 1)
        img_pad_ratios = img_pad_ratios.view(batch_size * beam_size, -1)

    end_flags = torch.zeros(batch_size, beam_size, dtype=torch.bool).to(device)

    for i in range(1, max_len - 1):
        # Prepare input
        input_word = ys[:, :, -1]  # (batch_size, beam_size)
        input_word = input_word.view(batch_size * beam_size, 1)  # (batch_size * beam_size, 1)

        # Decode with recurrent states
        y_n, s_ns = model.decode_recurrent(
            input_word, s_n_1s, k_img_list, v_img_list, i, img_pad_ratios
        )

        logits = model.generator(y_n)  # (batch_size * beam_size, 1, vocab_size)
        log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size * beam_size, vocab_size)

        # Calculate total scores
        scores = scores.view(batch_size * beam_size, 1)
        total_scores = scores + log_probs  # (batch_size * beam_size, vocab_size)

        # Reshape total_scores to (batch_size, beam_size * vocab_size)
        total_scores = total_scores.view(batch_size, beam_size * logits.size(-1))

        # Select top beam_size sequences
        next_scores, next_positions = total_scores.topk(beam_size, dim=1)  # (batch_size, beam_size)
        beam_indices = next_positions // logits.size(-1)  # Indices of previous beams
        token_indices = next_positions % logits.size(-1)  # Indices of next tokens

        # Update sequences ys
        ys = ys.gather(
            1, beam_indices.unsqueeze(-1).expand(-1, -1, ys.size(-1))
        )  # Select previous beams
        ys = torch.cat([ys, token_indices.unsqueeze(2)], dim=2)  # Append new tokens

        # Update scores
        scores = next_scores  # (batch_size, beam_size)

        # Update recurrent states
        def update_states(s_old, s_new):
            updated_s = []
            for old_s, new_s in zip(s_old, s_new):
                # Reshape new_s to (batch_size, beam_size, ...)
                new_s = new_s.view(batch_size, beam_size, *new_s.shape[1:])
                # Select beams
                selected_s = new_s.gather(
                    1,
                    beam_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *new_s.shape[2:]),
                )
                # Reshape back to (batch_size * beam_size, ...)
                selected_s = selected_s.view(batch_size * beam_size, *new_s.shape[2:])
                updated_s.append(selected_s)
            return updated_s

        s_n_1s = update_states(s_n_1s, s_ns)

        # Update end flags
        end_flags = end_flags.gather(1, beam_indices) | (token_indices == end_symbol)
        if end_flags.all():
            break

    # Select the best sequence from the beams
    best_scores, best_indices = scores.max(dim=1)
    best_sequences = ys[torch.arange(batch_size), best_indices, :]

    # Replace tokens after end_symbol with pad_idx
    for idx in range(batch_size):
        seq = best_sequences[idx]
        eos_indices = (seq == end_symbol).nonzero(as_tuple=False)
        if eos_indices.size(0) > 0:
            first_eos_idx = eos_indices[0].item()
            seq[first_eos_idx + 1 :] = pad_idx

    return best_sequences

def greedy_decode_recurrent_retnorm(model, src,img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    src = src.to(device) # (2,1,384,384)
    img_patches = model.feature_extractor(src) # (2,577,384)

    # Initialize ys with batch_size x 1, filled with start_symbol
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device) #(2,1)

    s_n_1s = [
            torch.zeros(batch_size,
                        model.heads,
                        model.d_model // model.heads,
                        model.d_model // model.heads).to(device) # (2,4,64,64)
        for _ in range(model.depth) #12
    ]
    k_prev_list = [None for _ in range(model.depth)]
    k_img_list, v_img_list = model.get_KV_img(img_patches, img_pad_ratios)
    # print("layers : ", len(x_0_to_n_1_s) )
    for i in range(max_len - 1): # 0~93
        input_word = ys[:, -1].unsqueeze(1) #(2,1)
        # input_word_embedded = model.embed_tokens(input_word) #(2,1,512)
        # b, n, _ = input_word_embedded.shape
        y_n, s_ns, k_prev_list = model.decode_recurrent_retnorm(input_word, s_n_1s, k_img_list, v_img_list, k_prev_list, i, img_pad_ratios)
        s_n_1s = s_ns
        # out = model.decode(ys, memory, pad_idx)
        logit = model.generator(y_n) #(2,1,79)
        # print(logit.shape)
        _, next_words = torch.max(logit, dim=2) #(2,1)

        # Check for EOS_IDX for each sequence in the batch
        eos_mask = next_words == end_symbol #(2,1)

        # Update ys with the next words
        ys = torch.cat([ys, next_words], dim=1) #(2,2) --> (2,3) --> .... --> (2,max_length)
        # If all sequences in the batch have reached EOS_IDX, break the loop
        if eos_mask.all():
            break
    # Replace the elements after EOS_IDX with pad_idx for each sequence
    for idx in range(batch_size):
        eos_idx = (ys[idx] == end_symbol).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            ys[idx, eos_idx[0] + 1:] = pad_idx

    return ys

def beam_decode_recurrent_retnorm(model, src, img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, beam_width, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    beam_size = beam_width
    src = src.to(device)
    batch_size = src.size(0)
    img_patches = model.feature_extractor(src)  # (batch_size, seq_len, model_dim)

    # Initialize sequences with the start symbol
    ys = torch.ones(batch_size, 1, 1).fill_(start_symbol).long().to(device)  # (batch_size, beam_size=1, seq_len=1)
    scores = torch.zeros(batch_size, 1).to(device)  # (batch_size, beam_size=1)

    # Initialize recurrent states for each layer
    s_n_1s = [
        torch.zeros(batch_size, 1, model.heads, model.d_model // model.heads, model.d_model // model.heads).to(device)
        for _ in range(model.depth)
    ]  # Each element: (batch_size, beam_size=1, heads, head_dim, v_dim_per_head)

    # Obtain key and value tensors from the image patches
    k_img_list, v_img_list = model.get_KV_img(img_patches, img_pad_ratios)
    k_prev_list = [None for _ in range(model.depth)]

    # First decoding step
    input_word = ys[:, 0, -1].unsqueeze(1)  # (batch_size, 1)
    s_n_1s_flat = [
        s.view(batch_size * 1, model.heads, model.d_model // model.heads, model.d_model // model.heads)
        for s in s_n_1s
    ]
    y_n, s_ns, k_prev_list = model.decode_recurrent_retnorm(
        input_word, s_n_1s_flat, k_img_list, v_img_list, k_prev_list, 0, img_pad_ratios
    )
    logits = model.generator(y_n)  # (batch_size, 1, vocab_size)
    log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size, vocab_size)

    # Select top beam_size tokens
    next_scores, next_tokens = log_probs.topk(beam_size, dim=1)  # (batch_size, beam_size)
    ys = next_tokens.unsqueeze(2)  # (batch_size, beam_size, seq_len=1)
    scores = next_scores  # (batch_size, beam_size)

    # Expand recurrent states to match beam_size
    def expand_states(s_list):
        expanded_list = []
        for s in s_list:
            s = s.unsqueeze(1)  # Shape: (batch_size, 1, heads, head_dim, v_dim_per_head)
            s = s.repeat(1, beam_size, 1, 1, 1, 1)  # (batch_size, beam_size, heads, head_dim, v_dim_per_head)
            s = s.view(
                batch_size * beam_size,
                model.heads,
                model.d_model // model.heads,
                model.d_model // model.heads,
            )
            expanded_list.append(s)
        return expanded_list

    s_n_1s = expand_states(s_ns)

    # Expand k_prev_list and v_prev_list to match beam_size
    def expand_kv_list(kv_list):
        expanded_list = []
        for kv in kv_list:
            kv = kv.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)  # (batch_size, beam_size, heads, seq_len, head_dim)
            kv = kv.view(batch_size * beam_size, *kv.shape[2:])
            expanded_list.append(kv)
        return expanded_list

    k_img_list = expand_kv_list(k_img_list)
    v_img_list = expand_kv_list(v_img_list)
    k_prev_list = expand_kv_list(k_prev_list)

    # Expand img_pad_ratios to match beam_size
    if img_pad_ratios is not None:
        img_pad_ratios = img_pad_ratios.unsqueeze(1).repeat(1, beam_size, 1)
        img_pad_ratios = img_pad_ratios.view(batch_size * beam_size, -1)

    end_flags = torch.zeros(batch_size, beam_size, dtype=torch.bool).to(device)

    for i in range(1, max_len - 1):
        # Prepare input
        input_word = ys[:, :, -1]  # (batch_size, beam_size)
        input_word = input_word.view(batch_size * beam_size, 1)  # (batch_size * beam_size, 1)

        # Decode with recurrent states
        y_n, s_ns, k_prev_list = model.decode_recurrent_retnorm(
            input_word, s_n_1s, k_img_list, v_img_list, k_prev_list, i, img_pad_ratios
        )

        logits = model.generator(y_n)  # (batch_size * beam_size, 1, vocab_size)
        log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size * beam_size, vocab_size)

        # Calculate total scores
        scores = scores.view(batch_size * beam_size, 1)
        total_scores = scores + log_probs  # (batch_size * beam_size, vocab_size)

        # Reshape total_scores to (batch_size, beam_size * vocab_size)
        total_scores = total_scores.view(batch_size, beam_size * logits.size(-1))

        # Select top beam_size sequences
        next_scores, next_positions = total_scores.topk(beam_size, dim=1)  # (batch_size, beam_size)
        beam_indices = next_positions // logits.size(-1)  # Indices of previous beams
        token_indices = next_positions % logits.size(-1)  # Indices of next tokens

        # Update sequences ys
        ys = ys.gather(
            1, beam_indices.unsqueeze(-1).expand(-1, -1, ys.size(-1))
        )  # Select previous beams
        ys = torch.cat([ys, token_indices.unsqueeze(2)], dim=2)  # Append new tokens

        # Update scores
        scores = next_scores  # (batch_size, beam_size)

        # Update recurrent states
        def update_states(s_old, s_new):
            updated_s = []
            for old_s, new_s in zip(s_old, s_new):
                # Reshape new_s to (batch_size, beam_size, ...)
                new_s = new_s.view(batch_size, beam_size, *new_s.shape[1:])
                # Select beams
                selected_s = new_s.gather(
                    1,
                    beam_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *new_s.shape[2:]),
                )
                # Reshape back to (batch_size * beam_size, ...)
                selected_s = selected_s.view(batch_size * beam_size, *new_s.shape[2:])
                updated_s.append(selected_s)
            return updated_s

        s_n_1s = update_states(s_n_1s, s_ns)

        # Update end flags
        end_flags = end_flags.gather(1, beam_indices) | (token_indices == end_symbol)
        if end_flags.all():
            break

    # Select the best sequence from the beams
    best_scores, best_indices = scores.max(dim=1)
    best_sequences = ys[torch.arange(batch_size), best_indices, :]

    # Replace tokens after end_symbol with pad_idx
    for idx in range(batch_size):
        seq = best_sequences[idx]
        eos_indices = (seq == end_symbol).nonzero(as_tuple=False)
        if eos_indices.size(0) > 0:
            first_eos_idx = eos_indices[0].item()
            seq[first_eos_idx + 1 :] = pad_idx

    return best_sequences

def greedy_decode_vanilla(model, src, img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    src = src.to(device)
    img_patches = model.feature_extractor(src)

    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(ys, img_patches, img_pad_ratios)
        logit = model.generator(out[:, -1, :])
        _, next_words = torch.max(logit, dim=1)

        eos_mask = next_words == end_symbol
        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        if eos_mask.all():
            break
    for idx in range(batch_size):
        eos_idx = (ys[idx] == end_symbol).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            ys[idx, eos_idx[0] + 1:] = pad_idx

    return ys
def beam_decode_vanilla(model, src, img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, beam_width, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    beam_size = beam_width
    src = src.to(device)
    batch_size = src.size(0)
    img_patches = model.feature_extractor(src)

    # Initialize sequences with the start symbol
    ys = torch.ones(batch_size, 1, 1).fill_(start_symbol).type(torch.long).to(device)  # (batch_size, beam_size=1, seq_len=1)
    scores = torch.zeros(batch_size, 1).to(device)  # (batch_size, beam_size=1)

    # First decoding step
    ys_flat = ys.view(batch_size * 1, -1)
    out = model.decode(ys_flat, img_patches, img_pad_ratios)
    logits = model.generator(out[:, -1, :])  # (batch_size, vocab_size)
    log_probs = F.log_softmax(logits, dim=1)  # (batch_size, vocab_size)

    # Select top beam_size beams
    next_scores, next_tokens = log_probs.topk(beam_size, dim=1)  #prob, indices : (batch_size, beam_size)
    ys = next_tokens.unsqueeze(2)  # (batch_size, beam_size, seq_len=1) # frist n candidates
    scores = next_scores  # (batch_size, beam_size)

    # Expand memory and img_pad_ratios to match beam_size
    img_patches = img_patches.unsqueeze(1).expand(-1, beam_size, -1, -1).contiguous()
    img_patches = img_patches.view(batch_size * beam_size, img_patches.size(2), img_patches.size(3))
    if img_pad_ratios is not None:
        img_pad_ratios = img_pad_ratios.unsqueeze(1).expand(-1, beam_size, -1).contiguous()
        img_pad_ratios = img_pad_ratios.view(batch_size * beam_size, -1)

    end_flags = torch.zeros(batch_size, beam_size, dtype=torch.bool).to(device)

    for i in range(1, max_len - 1):
        ys_flat = ys.view(batch_size * beam_size, -1)
        out = model.decode(ys_flat, img_patches, img_pad_ratios)
        logits = model.generator(out[:, -1, :])  # (batch_size * beam_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=1)

        # Accumulate scores
        scores = scores.view(batch_size * beam_size, 1) # (batch_size * beam_size,1)
        total_scores = scores + log_probs  # (batch_size * beam_size, vocab_size)
        total_scores = total_scores.view(batch_size, beam_size * logits.size(1)) # (batch_size, beam_size * vocab_size)

        # Select top beam_size sequences, logits.size(1) is vocab_size
        next_scores, next_scores_indices = total_scores.topk(beam_size, dim=1)
        # prev_beam_indices = next_scores_indices // logits.size(1) # warning : UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        prev_beam_indices = torch.div(next_scores_indices, logits.size(1), rounding_mode='trunc') # (batch_size, beam_size)
        next_tokens = next_scores_indices % logits.size(1) # (batch_size, beam_size)

        # Gather previous sequences
        ys = ys.view(batch_size, beam_size, -1)
        #.gather collects values from a tensor based on specified indices
        # this is just reordering
        selected_ys = ys.gather(1, prev_beam_indices.unsqueeze(-1).expand(-1, -1, ys.size(-1)))
        ys = torch.cat([selected_ys, next_tokens.unsqueeze(2)], dim=2)

        # Update scores and end flags
        scores = next_scores
        end_flags = end_flags.gather(1, prev_beam_indices) | (next_tokens == end_symbol)

        if end_flags.all():
            break

    # Select the best sequence from the beams
    best_scores, best_indices = scores.max(dim=1)
    best_sequences = ys[torch.arange(batch_size), best_indices, :]

    # Replace tokens after end_symbol with pad_idx
    for idx in range(batch_size):
        seq = best_sequences[idx]
        eos_indices = (seq == end_symbol).nonzero(as_tuple=False)
        if eos_indices.size(0) > 0:
            first_eos_idx = eos_indices[0].item()
            seq[first_eos_idx + 1:] = pad_idx

    return best_sequences
def greedy_decode_kv_cached(model, src,img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, device): #KV-cached
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    src = src.to(device) # (2,1,384,384)
    img_patches = model.feature_extractor(src) # (2,577,384)

    # Initialize ys with batch_size x 1, filled with start_symbol
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device) #(2,1)

    k_prev_list, v_prev_list = model.get_KV_img(img_patches, img_pad_ratios)
    for i in range(max_len - 1): # 0~93
        input_word = ys[:, -1].unsqueeze(1) #(2,1)
        y_n, k_prev_list, v_prev_list = model.decode_effective(input_word, img_patches, k_prev_list, v_prev_list, i, img_pad_ratios)
        logit = model.generator(y_n) #(2,1,79)
        _, next_words = torch.max(logit, dim=2) #(2,1)

        # Check for EOS_IDX for each sequence in the batch
        eos_mask = next_words == end_symbol #(2,1)

        # Update ys with the next words
        ys = torch.cat([ys, next_words], dim=1) #(2,2) --> (2,3) --> .... --> (2,max_length)
        # If all sequences in the batch have reached EOS_IDX, break the loop
        if eos_mask.all():
            break
    # Replace the elements after EOS_IDX with pad_idx for each sequence
    for idx in range(batch_size):
        eos_idx = (ys[idx] == end_symbol).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            ys[idx, eos_idx[0] + 1:] = pad_idx

    return ys
def beam_decode_kv_cached(model, src, img_pad_ratios, max_len, start_symbol, end_symbol, pad_idx, beam_width, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    beam_size = beam_width
    src = src.to(device)  # (batch_size, channels, height, width)
    batch_size = src.size(0)
    img_patches = model.feature_extractor(src)  # (batch_size, seq_len, model_dim)

    # Initialize sequences with the start symbol
    ys = torch.ones(batch_size, 1, 1).fill_(start_symbol).long().to(device)  # (batch_size, beam_size=1, seq_len=1)
    scores = torch.zeros(batch_size, 1).to(device)  # (batch_size, beam_size=1)

    # Obtain initial key and value caches from the image patches
    k_prev_list, v_prev_list = model.get_KV_img(img_patches, img_pad_ratios)
    # These are lists of tensors, one per layer

    # First decoding step
    input_word = ys[:, 0, -1].unsqueeze(1)  # (batch_size, 1)
    y_n, k_prev_list_new, v_prev_list_new = model.decode_effective(
        input_word, img_patches, k_prev_list, v_prev_list, 0, img_pad_ratios
    )
    logits = model.generator(y_n)  # (batch_size, 1, vocab_size)
    log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size, vocab_size)

    # Select top beam_size tokens
    next_scores, next_tokens = log_probs.topk(beam_size, dim=1)  # (batch_size, beam_size)
    ys = next_tokens.unsqueeze(2)  # (batch_size, beam_size, seq_len=1)
    scores = next_scores  # (batch_size, beam_size)

    # Expand caches to match beam_size
    def expand_cache(cache_list):
        expanded_list = []
        for cache in cache_list:
            cache = cache.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)  # (batch_size, beam_size, num_heads, seq_len, head_dim)
            cache = cache.view(batch_size * beam_size, *cache.shape[2:])  # (batch_size * beam_size, num_heads, seq_len, head_dim)
            expanded_list.append(cache)
        return expanded_list

    k_prev_list = expand_cache(k_prev_list_new)
    v_prev_list = expand_cache(v_prev_list_new)

    # Expand img_patches and img_pad_ratios to match beam_size
    img_patches = img_patches.unsqueeze(1).repeat(1, beam_size, 1, 1)
    img_patches = img_patches.view(batch_size * beam_size, *img_patches.shape[2:])
    if img_pad_ratios is not None:
        img_pad_ratios = img_pad_ratios.unsqueeze(1).repeat(1, beam_size, 1)
        img_pad_ratios = img_pad_ratios.view(batch_size * beam_size, -1)

    end_flags = torch.zeros(batch_size, beam_size, dtype=torch.bool).to(device)

    for i in range(1, max_len - 1):
        # Prepare input
        input_word = ys[:, :, -1]  # (batch_size, beam_size)
        input_word = input_word.view(batch_size * beam_size, 1)  # (batch_size * beam_size, 1)

        # Decode with key-value caching
        y_n, k_prev_list_new, v_prev_list_new = model.decode_effective(
            input_word, img_patches, k_prev_list, v_prev_list, i, img_pad_ratios
        )
        logits = model.generator(y_n)  # (batch_size * beam_size, 1, vocab_size)
        log_probs = F.log_softmax(logits, dim=2).squeeze(1)  # (batch_size * beam_size, vocab_size)

        # Calculate total scores
        scores = scores.view(batch_size * beam_size, 1)
        total_scores = scores + log_probs  # (batch_size * beam_size, vocab_size)

        # Reshape to (batch_size, beam_size * vocab_size)
        total_scores = total_scores.view(batch_size, beam_size * logits.size(-1))

        # Select top beam_size sequences
        next_scores, next_positions = total_scores.topk(beam_size, dim=1)  # (batch_size, beam_size)
        # prev_beam_indices = next_positions // logits.size(-1)  # Indices of previous beams
        prev_beam_indices = torch.div(next_positions, logits.size(-1), rounding_mode='trunc') #(batch_size, beam_size)
        token_indices = next_positions % logits.size(-1)  # Indices of next tokens #(batch_size, beam_size)

        # Update sequences ys : # this is just reordering
        ys = ys.gather(1, prev_beam_indices.unsqueeze(-1).expand(-1, -1, ys.size(-1)))  # (batch_size, beam_size, seq_len)
        ys = torch.cat([ys, token_indices.unsqueeze(2)], dim=2)  # Append new tokens

        # Update scores
        scores = next_scores

        # Update caches
        def update_cache(cache_old, cache_new):
            updated_cache = []
            for c_old, c_new in zip(cache_old, cache_new):
                # Reshape to (batch_size, beam_size, ...)
                c_new = c_new.view(batch_size, beam_size, *c_new.shape[1:]) # (batch_size, beam_size, num_heads, seq_len, head_dim)
                # Select beams # only reorders the elements in the beam_size dimension according to the indices specified in prev_beam_indices
                selected_c = c_new.gather(
                    1,
                    prev_beam_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *c_new.shape[2:])
                )
                # Reshape back to (batch_size * beam_size, ...)
                selected_c = selected_c.view(batch_size * beam_size, *c_new.shape[2:])
                updated_cache.append(selected_c)
            return updated_cache

        k_prev_list = update_cache(k_prev_list, k_prev_list_new)
        v_prev_list = update_cache(v_prev_list, v_prev_list_new)

        # Update end flags
        end_flags = end_flags.gather(1, prev_beam_indices) | (token_indices == end_symbol)
        if end_flags.all():
            break

    # Select the best sequence from the beams
    best_scores, best_indices = scores.max(dim=1)
    best_sequences = ys[torch.arange(batch_size), best_indices, :]

    # Replace tokens after end_symbol with pad_idx
    for idx in range(batch_size):
        seq = best_sequences[idx]
        eos_indices = (seq == end_symbol).nonzero(as_tuple=False)
        if eos_indices.size(0) > 0:
            first_eos_idx = eos_indices[0].item()
            seq[first_eos_idx + 1:] = pad_idx

    return best_sequences

