from llama import ModelArgs, Transformer, Tokenizer
import torch
import os
import click
import json
import numpy as np
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from arithmeticcoding_fast import ArithmeticEncoder, BitOutputStream


def setup_model_parallel():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print("Local Rank : ",local_rank,", World Size : ",world_size)

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    torch.manual_seed(1)
    return local_rank, world_size



class LLMComressor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(
        self,
        win_size,
        tokens_full,
        dir_to_save,
    ):
        win_size_enc = win_size + 1 
        
        with open(os.path.join(dir_to_save, 'comressed.txt'), 'wb') as f:

            bitout = BitOutputStream(f)
            self.encoder = ArithmeticEncoder(32, bitout)

            bsz = 1             
            ranks_list = []
            probs_tok_list = []

            n_runs = tokens_full.size-win_size_enc+1

            for t_ind in range(1,win_size_enc):
                tokens_in = np.array([[self.tokenizer.bos_id]+tokens_full[:t_ind].tolist()])
                ranks,probs_tok = self.encode_batch(tokens_in)
                ranks_list += [ranks]
                probs_tok_list += [probs_tok]

            
            n_batches = np.ceil(n_runs/bsz).astype(int)

            for b_ind in range(n_batches):

                batch_range_start = b_ind*bsz
                batch_range_stop = np.minimum(n_runs,(b_ind+1)*bsz)
                tokens_batch = np.array([tokens_full[i:i+win_size_enc]for i in range(batch_range_start,batch_range_stop)])
                ranks,probs_tok = self.encode_batch(tokens_batch)
                ranks_list += [ranks]
                probs_tok_list += [probs_tok]
                
                
                if (b_ind*bsz*100/n_batches)%10 == 0:
                    print(f'Encoder: Completed {int(b_ind*bsz*100/n_batches)} %')
                
            
            self.encoder.finish()
            bitout.close()


    def encode_batch(
        self,
        prompt_tokens
    ):
        
        bsz = prompt_tokens.shape[0]           
        prompt_size = prompt_tokens.shape[1]

        tokens = torch.full((bsz, prompt_size), self.tokenizer.pad_id).cuda().long()
        tokens[:bsz, : prompt_size] = torch.tensor(prompt_tokens).long()

        cur_pos = prompt_size-1
        prev_pos = 0
        
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits, dim=-1)
        rank = gen_rank(probs,next_token=tokens[:,cur_pos])
        
        probs_np2 = probs.cpu().numpy()
        tokens_np2 = tokens[:,cur_pos].cpu().numpy()
        ranks_np2 = rank.cpu().numpy()
        
        probs_tok = probs_np2[np.arange(bsz),tokens_np2]
        

        cumul = np.zeros(self.model.vocab_size+1, dtype = np.uint64)
        for j in range(bsz):
            prob1 = probs_np2[j]
            cumul[1:] = np.cumsum(prob1*10000000 + 1)
            self.encoder.write(cumul, tokens_np2[j])
        
        return ranks_np2,probs_tok
        

def gen_rank(probs,next_token):
    _, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True) 
    rank_list = []
    if next_token.shape[0]>1:
        for i in range(next_token.shape[0]):
            rank_list += [torch.where(probs_idx[i:i+1,:] == next_token[i])[-1]]
        rank = torch.squeeze(torch.stack(rank_list))
    else:
        rank = torch.where(probs_idx == next_token)[-1]
    return rank




@click.command()
@click.option("-s", "model_path", required=True, help="Directory with tokenizer and model weights")
@click.option("-f", "text_file", required=True, help="Path to file to be compressed")
@click.option("-w", "win_len", default=64, help="Window length")
@click.option("-d", "target_directory",required=True, help="Directory to save compressed file")

def main(
    model_path: str,
    text_file: str, 
    win_len: int,
    target_directory: str,
):

    setup_model_parallel()

    max_seq_len = 512
    max_batch_size = 32

    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    checkpoint_path = f"{model_path}/7B/"

    checkpoint = torch.load(os.path.join(checkpoint_path, "consolidated.00.pth"))
    with open(os.path.join(checkpoint_path, "params.json"), "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.vocab_size=tokenizer.n_words


    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    
    # torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    os.makedirs(target_directory, exist_ok=True)

    with open(text_file,'r') as f:
        source_text = f.read()

    tokens = np.array(tokenizer.encode(source_text, bos=False, eos=False))

    compressor = LLMComressor(model, tokenizer)
    compressor.encode(win_len, tokens, target_directory)



if __name__ == "__main__":
    main()