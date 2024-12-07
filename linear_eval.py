def linear_eval():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = MoCo().cuda()
    model.load_state_dict(torch.load('moco_final.pt'))
    model.eval()

    classifier = nn.Sequential(
        nn.BatchNorm1d(128),
        nn.Linear(128, 10)
    ).cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        classifier.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                features = model.encoder_q(images)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()

            features = model.encoder_q(images)
            outputs = classifier(features)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return acc

if __name__ == "__main__":
    linear_eval()
