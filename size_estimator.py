import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from colossalai.utils import print_rank_0

class SizeEstimator(object):

    def __init__(self, model, batch, real_bs, bytes=2, bytes_input=4, 
                 gpu_n=1, tp=0, lm_fp32=False, m_total=0):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.batch = batch
        self.bytes = bytes
        self.bytes_input = bytes_input
        self.gpu_n = gpu_n
        self.tp = tp
        self.base_size = 2*1024*1024 # 512
        self.real_bs = real_bs
        self.lm_fp32 = lm_fp32
        self.m_total = m_total
        

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        # input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        input_ = self.batch
        cnt = 0
        pr_out = 0
        repeat_layer = 0
        embed_cnt = 0
        block_first = 0
        for name, _ in self.model.named_modules():
            if 'project_out' in name:
                pr_out = cnt
                break
            cnt += 1
        mods = list(self.model.modules())
        inout_sizes = [np.array(input_.size())]
        backward_all_gather_sizes = []
        for i in range(1, len(mods)):
            if 'DecoderLayer' in mods[i]._get_name(): #or 'BertLayer' 
                if repeat_layer > 0:
                    inout_sizes.append(np.array(input_.size()))
                    # temp_size = np.array(input_.size())
                    # temp_size[2] *= 4
                    backward_all_gather_sizes.append(np.array(input_.size()))
                repeat_layer += 1
            elif 'Block' in mods[i]._get_name(): # 'BloomBlock' or 'CodeGenBlock'
                if repeat_layer > 0:
                    inout_sizes.append(np.array(input_.size()))
                    # temp_size = np.array(input_.size())
                    # temp_size[2] *= 4
                    backward_all_gather_sizes.append(np.array(input_.size()))
                repeat_layer += 1
                block_first = 1
            if not mods[i]._get_name() in ['Embedding', 'Linear']: #, 'ReLU', 'LayerNorm', 'GELUActivation']:
                continue
            if i == pr_out:
                continue
            ##################### For BERT #####################
            temp_embed = 0
            if mods[i]._get_name() in ['Embedding']:
                embed_cnt += 1
                temp_embed = 1
            if temp_embed == 1 and embed_cnt > 1:
                continue
            ####################################################
            ##################### For Block #####################
            if mods[i]._get_name() in ['Linear']:
                if block_first == 1:
                    block_first = 0
                    continue
            #####################################################
            if pr_out != 0 and i == len(mods) - 1:
                m = mods[pr_out]
                out = m(input_)
                inout_sizes.append(np.array(out.size()))
                backward_all_gather_sizes.append(np.array(out.size()))
                input_ = out
                m = mods[i]
                out = m(input_)
                inout_sizes.append(np.array(out.size()))
                backward_all_gather_sizes.append(np.array(out.size()))
                break
            m = mods[i]
            out = m(input_)
            if mods[i]._get_name() in ['Embedding']:
                inout_sizes.append(np.array(out.size()))    # embed_tokens
                inout_sizes.append(np.array(out.size()))    # embed_positions
            if i == len(mods) - 1:
                inout_sizes.append(inout_sizes[-1])         # last layer of ModuleList
                inout_sizes.append(np.array(out.size()))    # lm_head
                backward_all_gather_sizes.append(backward_all_gather_sizes[-1])
            input_ = out

        self.inout_sizes = inout_sizes
        self.backward_all_gather_sizes = backward_all_gather_sizes


    def param_bytes(self):
        mods = list(self.model.modules())
        param_sizes =[]
        for i in range(1, len(mods)):
            if not 'Embedding' in mods[i]._get_name():
               if not mods[i]._get_name() in ['Linear']: #, 'ReLU', 'LayerNorm']:
                    continue
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                param_sizes.append(np.array(p[j].size()))

        '''Calculate total number of bytes to store `model` parameters'''
        total_bytes = 0
        for i in range(len(param_sizes)):
            s = param_sizes[i]
            bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            if i == len(param_sizes) - 1:
                # break
                total_bytes += bytes# / self.gpu_n
            else:
                # Chunk-based mixed-precision model parameter memory
                # 1. Parameters/gradients (fp16)
                if not self.tp:
                    if self.gpu_n > 1:  # ZeRO-3 Optimizer
                        # total_bytes += bytes
                        bytes = bytes * (self.gpu_n - 1) / self.gpu_n
                        if bytes % self.base_size != 0:
                            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                        total_bytes += bytes
                elif self.tp and self.gpu_n > 1 and self.tp != self.gpu_n: # DP + TP (Hybrid parallelism)
                    bytes = bytes * ((self.gpu_n - self.tp) / self.gpu_n - 1 / self.gpu_n)
                    if bytes % self.base_size != 0:
                        bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                    total_bytes += bytes
                # 2. optimizer parameters (fp32)
                bytes = np.prod(np.array(s))*self.bytes*2
                bytes = bytes / self.gpu_n
                if bytes % self.base_size != 0:
                    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                ##### Real-size-based optimizer states memory
                # 3. gradient momentums (fp32), gradient variances (fp32)
                total_bytes += 2*bytes
                ##########################
        self.param_bytes = total_bytes


    def calc_input_bytes(self):
        '''Calculate bytes to store input'''
        # self.input_bytes = np.prod(np.array(self.input_size))*self.bytes
        self.input_bytes = np.prod(self.inout_sizes[0])*self.bytes_input
        if self.input_bytes % self.base_size != 0:
            self.input_bytes = int(self.input_bytes / self.base_size) * self.base_size + self.base_size


    def calc_output_bytes(self):
        '''Calculate bytes to store forward and backward pass'''
        total_bytes = 0
        total_backward_all_gather_bytes = 0
        for i in range(0, len(self.inout_sizes)):
            self.inout_sizes[i][0] = self.real_bs
        for i in range(1, len(self.inout_sizes)-1):
            s = self.inout_sizes[i]
            bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            total_bytes += bytes

        if self.tp:
            for i in range(0, len(self.backward_all_gather_sizes)):
                self.backward_all_gather_sizes[i][0] = self.real_bs
            for i in range(0, len(self.backward_all_gather_sizes)-1):
                s = self.backward_all_gather_sizes[i]
                bytes = np.prod(np.array(s))*self.bytes * (self.tp - 1) / self.tp
                if bytes % self.base_size != 0:
                    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                total_backward_all_gather_bytes += bytes

        # lm_head and loss function
        last_part = 0
        s = self.inout_sizes[-1]
        if self.lm_fp32:
            bytes = np.prod(np.array(s))*self.bytes*2
        else:
            bytes = np.prod(np.array(s))*self.bytes
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
        last_part += bytes
        s[1] -= 1
        # temporary_mem = 0
        if self.lm_fp32:
            bytes = np.prod(np.array(s))*self.bytes*2
            # temporary_mem = bytes / 2
        else:
            bytes = np.prod(np.array(s))*self.bytes
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
        # if temporary_mem % self.base_size != 0:
        #     temporary_mem = int(temporary_mem / self.base_size) * self.base_size + self.base_size
        last_part += bytes*2 # + temporary_mem

        if total_backward_all_gather_bytes > 0:
            last_part += total_backward_all_gather_bytes

        self.inout_bytes = total_bytes + last_part


    def bs_search(self):
        '''Find the maximum batch size'''
        cur_bs = 1
        last_total_mem = 0
        while True:
            total_bytes = 0
            total_backward_all_gather_bytes = 0
            total_mem = self.m_init
            for i in range(0, len(self.inout_sizes)):
                self.inout_sizes[i][0] = cur_bs
            for i in range(1, len(self.inout_sizes)-1):
                s = self.inout_sizes[i]
                bytes = np.prod(np.array(s))*self.bytes
                if bytes % self.base_size != 0:
                    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                total_bytes += bytes

            if self.tp:
                for i in range(0, len(self.backward_all_gather_sizes)):
                    self.backward_all_gather_sizes[i][0] = cur_bs
                for i in range(0, len(self.backward_all_gather_sizes)-1):
                    s = self.backward_all_gather_sizes[i]
                    bytes = np.prod(np.array(s))*self.bytes * (self.tp - 1) / self.tp
                    if bytes % self.base_size != 0:
                        bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                    total_backward_all_gather_bytes += bytes

            # lm_head and loss function
            last_part = 0
            s = self.inout_sizes[-1]
            if self.lm_fp32:
                bytes = np.prod(np.array(s))*self.bytes*2
            else:
                bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            last_part += bytes
            s[1] -= 1
            # temporary_mem = 0
            if self.lm_fp32:
                bytes = np.prod(np.array(s))*self.bytes*2
                # temporary_mem = bytes / 2
            else:
                bytes = np.prod(np.array(s))*self.bytes
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size
            # if temporary_mem % self.base_size != 0:
            #     temporary_mem = int(temporary_mem / self.base_size) * self.base_size + self.base_size
            last_part += bytes*2 #+ temporary_mem

            # if total_backward_all_gather_bytes > bytes:
            #     last_part += total_backward_all_gather_bytes - bytes
            if total_backward_all_gather_bytes > 0:
                last_part += total_backward_all_gather_bytes

            total_mem += (self.param_bytes + self.input_bytes + total_bytes + last_part) / (1024**2)

            # ####################################################
            # import torch.distributed as dist
            # if dist.get_rank() == 0:
            #     with open('temp_size.txt', 'a') as f:
            #         # f.write('{}: {}\n'.format(cur_bs, total_mem))
            #         f.write('{}\n'.format(total_mem))
            # ####################################################
                    
            if total_mem > self.m_total:
                self.real_bs = cur_bs - 1
                if last_total_mem == 0:
                    return total_mem
                return last_total_mem
            # if cur_bs > 80:
            #     return last_total_mem
            
            cur_bs += 1
            last_total_mem = total_mem
            s[1] += 1

    # def calc_temp_buf_bytes(self):
    #     '''Calculate bytes to keep temporary buffer for intermediate results'''
    #     mods = list(self.model.modules())
    #     sizes = []
    #     for i in range(1, len(mods)):
    #         if not mods[i]._get_name() in ['Linear']:
    #             continue
    #         m = mods[i]
    #         p = list(m.parameters())
    #         for j in range(len(p)):
    #             sizes.append(np.array(p[j].size()))

    def estimate_size(self, m_init=0):
        '''Estimate model size in memory in megabytes and bytes'''
        self.m_init = m_init
        self.param_bytes()
        # self.get_output_sizes()
        self.calc_input_bytes()

        if self.real_bs > 0:
            self.calc_output_bytes()
            total = self.param_bytes + self.inout_bytes + self.input_bytes
            total_mb = total/(1024**2) + self.m_init
            return total_mb, self.real_bs
        else:
            total_mb = self.bs_search()
            return total_mb, self.real_bs
        
        