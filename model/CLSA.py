import torch
import torch.nn as nn

class CLSA(nn.Module):

    def __init__(self, base_encoder, args, dim=128, K=65536, m=0.999, T=0.2, mlp=True):
        """
        :param base_encoder: encoder model
        :param args: config parameters
        :param dim: feature dimension (default: 128)
        :param K: queue size; number of negative keys (default: 65536)
        :param m: momentum of updating key encoder (default: 0.999)
        :param T: softmax temperature (default: 0.2)
        :param mlp: use MLP layer to process encoder output or not (default: True)
        """
        super(CLSA, self).__init__()
        self.args = args
        self.K = K
        self.m = m
        self.T = T
        self.T2 = self.args.clsa_t

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)  # normalize across queue instead of each example
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # config parameters for CLSA stronger augmentation and multi-crop
        self.weak_pick = args.pick_weak
        self.strong_pick = args.pick_strong
        self.weak_pick = set(self.weak_pick)
        self.strong_pick = set(self.strong_pick)
        self.gpu = args.gpu
        self.sym = self.args.sym

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue, queue_ptr, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys) #already concatenated before

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    def forward(self, im_q_list, im_k,im_strong_list):
        """
        :param im_q_list: query image list
        :param im_k: key image
        :param im_strong_list: query strong image list
        :return:
        weak: logit_list, label_list
        strong: logit_list, label_list
        """
        if self.sym:
            q_list = []
            for k, im_q in enumerate(im_q_list):  # weak forward
                if k not in self.weak_pick:
                    continue
                # can't shuffle because it will stop gradient only can be applied for k
                # im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)
                # q = self._batch_unshuffle_ddp(q, idx_unshuffle)
                q_list.append(q)
            # add the encoding of im_k as one of weakly supervised
            q = self.encoder_q(im_k)
            q = nn.functional.normalize(q, dim=1)
            q_list.append(q)

            q_strong_list = []
            for k, im_strong in enumerate(im_strong_list):
                # im_strong, idx_unshuffle = self._batch_shuffle_ddp(im_strong)
                if k not in self.strong_pick:
                    continue
                q_strong = self.encoder_q(im_strong)  # queries: NxC
                q_strong = nn.functional.normalize(q_strong, dim=1)
                # q_strong = self._batch_unshuffle_ddp(q_strong, idx_unshuffle)
                q_strong_list.append(q_strong)
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k = k.detach()
                k = concat_all_gather(k)

                k2 = self.encoder_k(im_q_list[0])  # keys: NxC
                k2 = nn.functional.normalize(k2, dim=1)
                # undo shuffle
                k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
                k2 = k2.detach()
                k2 = concat_all_gather(k2)
            logits0_list = []
            labels0_list = []
            logits1_list = []
            labels1_list = []
            # first iter the 1st k supervised
            for choose_idx in range(len(q_list) - 1):
                q = q_list[choose_idx]
                # positive logits: NxN
                l_pos = torch.einsum('nc,ck->nk', [q, k.T])
                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators

                cur_batch_size = logits.shape[0]
                cur_gpu = self.gpu
                choose_match = cur_gpu * cur_batch_size
                labels = torch.arange(choose_match, choose_match + cur_batch_size, dtype=torch.long).cuda()

                logits0_list.append(logits)
                labels0_list.append(labels)

                labels0 = logits.clone().detach()  # use previous q as supervision
                labels0 = labels0 * self.T / self.T2
                labels0 = torch.softmax(labels0, dim=1)
                labels0 = labels0.detach()
                for choose_idx2 in range(len(q_strong_list)):
                    q_strong = q_strong_list[choose_idx2]
                    # weak strong loss

                    l_pos = torch.einsum('nc,ck->nk', [q_strong, k.T])
                    # negative logits: NxK
                    l_neg = torch.einsum('nc,ck->nk', [q_strong, self.queue.clone().detach()])

                    # logits: Nx(1+K)
                    logits0 = torch.cat([l_pos, l_neg], dim=1)  # N*(K+1)

                    # apply temperature
                    logits0 /= self.T2
                    logits0 = torch.softmax(logits0, dim=1)

                    logits1_list.append(logits0)
                    labels1_list.append(labels0)
            # iter another part, symmetrized
            k = k2
            for choose_idx in range(1, len(q_list)):
                q = q_list[choose_idx]
                # positive logits: NxN
                l_pos = torch.einsum('nc,ck->nk', [q, k.T])
                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators

                cur_batch_size = logits.shape[0]
                cur_gpu = self.gpu
                choose_match = cur_gpu * cur_batch_size
                labels = torch.arange(choose_match, choose_match + cur_batch_size, dtype=torch.long).cuda()

                logits0_list.append(logits)
                labels0_list.append(labels)

                labels0 = logits.clone().detach()  # use previous q as supervision
                labels0 = labels0 * self.T / self.T2
                labels0 = torch.softmax(labels0, dim=1)
                labels0 = labels0.detach()
                for choose_idx2 in range(len(q_strong_list)):
                    q_strong = q_strong_list[choose_idx2]
                    # weak strong loss

                    l_pos = torch.einsum('nc,ck->nk', [q_strong, k.T])
                    # negative logits: NxK
                    l_neg = torch.einsum('nc,ck->nk', [q_strong, self.queue.clone().detach()])

                    # logits: Nx(1+K)
                    logits0 = torch.cat([l_pos, l_neg], dim=1)  # N*(K+1)

                    # apply temperature
                    logits0 /= self.T2
                    logits0 = torch.softmax(logits0, dim=1)

                    logits1_list.append(logits0)
                    labels1_list.append(labels0)

            # dequeue and enqueue
            # if update_key_encoder==False:
            self._dequeue_and_enqueue(self.queue, self.queue_ptr, k)

            return logits0_list, labels0_list, logits1_list, labels1_list
        else:
            q_list = []
            for k, im_q in enumerate(im_q_list):  # weak forward
                if k not in self.weak_pick:
                    continue
                # can't shuffle because it will stop gradient only can be applied for k
                # im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)
                # q = self._batch_unshuffle_ddp(q, idx_unshuffle)
                q_list.append(q)

            q_strong_list = []
            for k, im_strong in enumerate(im_strong_list):
                # im_strong, idx_unshuffle = self._batch_shuffle_ddp(im_strong)
                if k not in self.strong_pick:
                    continue
                q_strong = self.encoder_q(im_strong)  # queries: NxC
                q_strong = nn.functional.normalize(q_strong, dim=1)
                # q_strong = self._batch_unshuffle_ddp(q_strong, idx_unshuffle)
                q_strong_list.append(q_strong)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k = k.detach()
                k = concat_all_gather(k)

            # compute logits
            # Einstein sum is more intuitive

            logits0_list = []
            labels0_list = []
            logits1_list = []
            labels1_list = []
            for choose_idx in range(len(q_list)):
                q = q_list[choose_idx]

                # positive logits: Nx1
                l_pos = torch.einsum('nc,ck->nk', [q, k.T])
                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators
                cur_batch_size = logits.shape[0]
                cur_gpu = self.gpu
                choose_match = cur_gpu * cur_batch_size
                labels = torch.arange(choose_match, choose_match + cur_batch_size, dtype=torch.long).cuda()

                logits0_list.append(logits)
                labels0_list.append(labels)

                labels0 = logits.clone().detach()  # use previous q as supervision
                labels0 = labels0*self.T/self.T2
                labels0 = torch.softmax(labels0, dim=1)
                labels0 = labels0.detach()
                for choose_idx2 in range(len(q_strong_list)):
                    q_strong = q_strong_list[choose_idx2]
                    # weak strong loss

                    l_pos = torch.einsum('nc,ck->nk', [q_strong, k.T])
                    # negative logits: NxK
                    l_neg = torch.einsum('nc,ck->nk', [q_strong, self.queue.clone().detach()])

                    # logits: Nx(1+K)
                    logits0 = torch.cat([l_pos, l_neg], dim=1)  # N*(K+1)

                    # apply temperature
                    logits0 /= self.T2
                    logits0 = torch.softmax(logits0, dim=1)

                    logits1_list.append(logits0)
                    labels1_list.append(labels0)

            # dequeue and enqueue
            # if update_key_encoder==False:
            self._dequeue_and_enqueue(self.queue, self.queue_ptr, k)

            return logits0_list, labels0_list, logits1_list, labels1_list





@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
