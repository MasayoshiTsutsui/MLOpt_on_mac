import torch
import torchvision
import time

device = torch.device("cpu")
vgg = torchvision.models.vgg11(pretrained=True).to(device)

# run 10 times and measure the average time
num_runs = 100
total_time = 0

for _ in range(num_runs):
    # モデルを推論モードに設定
    vgg.eval()

    # 入力データ（仮想的なデータで初期化）
    inputs = torch.randn(1, 3, 224, 224).to(device)

    # 推論
    start_time = time.time()
    with torch.no_grad():
        outputs = vgg(inputs)
    end_time = time.time()

    # 実行時間を加算
    total_time += end_time - start_time

# 平均実行時間を計算
average_time = total_time / num_runs

print(f"Average inference time: {average_time:.5f} seconds")




h = 224
w = 224
c = 3
traced_vgg = torch.jit.trace(vgg, torch.randn(1, c, h, w).to(device))
traced_vgg.save("vgg11_h{}_w{}_c{}.pt".format(h, w, c))
print("vgg11_h{}_w{}_c{}.pt is exported".format(h, w, c))
