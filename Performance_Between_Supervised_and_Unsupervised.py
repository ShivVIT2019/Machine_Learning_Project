from torchvision.models import resnet18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

def supervised_baseline():
    # Data transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Datasets and DataLoader
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # ResNet-18 model
    model = resnet18(num_classes=10).cuda()

    # Optimizer, criterion, scheduler
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Training loop
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Supervised Baseline Accuracy: {100. * correct / total:.2f}%")






if __name__ == "__main__":
    supervised_baseline()

