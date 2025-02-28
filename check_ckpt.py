# import mindspore as ms
# import numpy as np

# infer_raw = ms.load_checkpoint("/grpo/hx0226/model_ckpt/qwen2_7b_241/qwen2_ms/rank_0/checkpoint_0.ckpt")
# infer_raw = {'grpo_model.policy_model.model.' + k: v for k, v in infer_raw.items()}
# # infer_old = ms.load_checkpoint("/grpo/hx0226/mindrlhf_2_0226/grpo_infer_ckpt_0_0_0.ckpt")
# # infer_new = ms.load_checkpoint("/grpo/hx0226/mindrlhf_2_0226/grpo_infer_ckpt_0_0_1.ckpt")

# for key, value in infer_raw.items():
#     if "_cache" not in key:
#         cc = infer_raw[key].asnumpy().astype(np.float32)
#     else:
#         cc = None
#     if "attention.wq.weight" in key:
#         print("attention.wq.weight.shape:", cc.shape)

#     # aa = infer_old[key].asnumpy().astype(np.float32)
#     # bb = infer_new[key].asnumpy().astype(np.float32)
#     # flag = np.allclose(aa, bb, rtol=1e-05, atol=1e-08, equal_nan=False)

#     # if not flag:
#     #     print("cc:", cc)
#     #     print("aa:", aa)
#     #     print("bb:", bb)
#     #     print(key, "not equal")
#     # else:
#     #     print(key, "equal")


import mindspore as ms
import numpy as np

infer_raw = ms.load_checkpoint("/grpo/hx0226/model_ckpt/qwen2_7b_241/qwen2_ms/rank_0/checkpoint_0.ckpt")
infer_raw = {'grpo_model.policy_model.model.' + k: v for k, v in infer_raw.items()}
infer_old = ms.load_checkpoint("/grpo/hx0226/mindrlhf_2_0226/grpo_infer_ckpt_0_0_0.ckpt")
infer_new = ms.load_checkpoint("/grpo/hx0226/mindrlhf_2_0226/grpo_infer_ckpt_0_0_1.ckpt")

for key, value in infer_old.items():
    if "_cache" not in key:
        cc = infer_raw[key].asnumpy().astype(np.float32)
    else:
        cc = None
    if "attention.wq.weight" in key:
        print("attention.wq.weight.shape:", cc.shape)

    aa = infer_old[key].asnumpy().astype(np.float32)
    bb = infer_new[key].asnumpy().astype(np.float32)
    flag = np.allclose(aa, bb, rtol=1e-05, atol=1e-08, equal_nan=False)

    if not flag:
        # print("cc:", cc)
        # print("aa:", aa)
        # print("bb:", bb)
        print(key, "not equal")
    # else:
    #     print(key, "equal")


