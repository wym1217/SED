import torch
import models.crnn4conv as models
print(torch.cuda.is_available()) 
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
print(torch.__version__)  # 查看 PyTorch 版本