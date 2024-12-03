def linear_eval():
    # Create test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 train and test sets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Load pretrained MoCo model
    model = MoCo().cuda()
    model.load_state_dict(torch.load('moco_final.pt'))
    model.eval()  # Set to evaluation mode

    # Create linear classifier with BatchNorm
    classifier = nn.Sequential(
        nn.BatchNorm1d(128),
        nn.Linear(128, 10)
    ).cuda()  # 128 is embedding dim, 10 for CIFAR-10 classes

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)  # 60 epochs
    criterion = nn.CrossEntropyLoss()

    # Train linear classifier for 60 epochs
    print("Training linear classifier...")
    for epoch in range(60):
        classifier.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Extract frozen features from MoCo encoder
            with torch.no_grad():
                features = model.encoder_q(images)

            # Forward pass through linear classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backpropagation and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

    # Save the classifier after training
    torch.save(classifier.state_dict(), 'classifier.pt')

    # Evaluate test accuracy
    print("Evaluating on test set...")
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            features = model.encoder_q(images)
            outputs = classifier(features)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'Accuracy on CIFAR-10 test set: {acc:.2f}%')
    return acc

if __name__ == "__main__":
    acc = linear_eval()
