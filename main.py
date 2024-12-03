import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt

# Hyperparameters
K = 4096
m = 0.99
T = 0.07
EPOCHS = 50
BATCH_SIZE = 256

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return nn.functional.normalize(x, dim=1)

class MoCo(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = Encoder(dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Debug prints
        # print(f"Pos sim: {l_pos.mean():.3f}, Neg sim: {l_neg.mean():.3f}")

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

def train_moco():
    model = MoCo().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.encoder_q.parameters(),
                               lr=0.03,
                               momentum=0.9,
                               weight_decay=1e-4)

    # Simple step scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.1)

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=TwoCropsTransform(transform_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, drop_last=True, pin_memory=True)

    train_losses = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        batch_count = 0
        for i, (images, _) in enumerate(dataloader):
            im_q, im_k = images[0].cuda(), images[1].cuda()

            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        scheduler.step()
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'moco_checkpoint_epoch_{epoch+1}.pt')

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.show()

    # Save final model
    torch.save(model.state_dict(), 'moco_final.pt')
    return train_losses

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

if __name__ == "__main__":
    losses = train_moco()
