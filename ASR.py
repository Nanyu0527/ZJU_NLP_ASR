import torch
import time

n = 4000
x = torch.ones(n,n)

start_time = time.time()

# 运行在CPU上
y = x.mm(x)

end_time = time.time()
print(f'Elapsed time on CPU: {(end_time - start_time)*1000:.2f} ms')

if torch.cuda.is_available():
    x = x.cuda()
    start_time = time.time()

    # 运行在GPU上
    y = x.mm(x)

    end_time = time.time()
    print(f'Elapsed time on GPU: {(end_time - start_time)*1000:.2f} ms')