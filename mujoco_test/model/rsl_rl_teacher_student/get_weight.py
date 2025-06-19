import torch

# 1. 加载原始 checkpoint
path = "/home/yd/program/rsl_rl_teacher_student/legged_gym/logs/rough_go2/Jun10_16-36-50_/model_9000.pt"
ckpt = torch.load(path, map_location="cpu")
model_dict = ckpt["model_state_dict"]

# 2. 提取 proprioceptive_encoder
prop_keys = [k for k in model_dict if k.startswith("proprioceptive_encoder.")]
prop_dict = {
    k.replace("proprioceptive_encoder.", ""): model_dict[k]  # 移除前缀
    for k in prop_keys
}

# 3. 提取 actor
actor_keys = [k for k in model_dict if k.startswith("actor.")]
actor_dict = {
    k: model_dict[k]  # 保留原始键名称
    for k in actor_keys
}

# 4. 保存为 .pth
torch.save(prop_dict, "proprio_oracle.pth")
print("✅ 已保存 proprio_oracle.pth")

torch.save(actor_dict, "actor_oracle.pth")
print("✅ 已保存 actor_oracle.pth")
