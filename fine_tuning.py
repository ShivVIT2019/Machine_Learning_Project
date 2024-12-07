class FineTunedMoCo(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_q = Encoder(dim=512)  # Use the original encoder architecture
        self.encoder_q.projection = nn.Sequential(
            nn.Linear(512, 10)  # Projection head for CIFAR-10 classification
        )

    def forward(self, x):
        return self.encoder_q(x)

def fine_tune_with_labels():
    print("Starting fine-tuning process...")
    # Load pretrained MoCo encoder
    model = MoCo().cuda()
    model.load_state_dict(torch.load('moco_final.pt'))

    # Unfreeze the last block of the encoder and projection head
    for param in model.encoder_q.encoder[-1].parameters():
        param.requires_grad = True
    for param in model.encoder_q.projection.parameters():
        param.requires_grad = True

    # Replace projection head with a better classifier
    model.encoder_q.projection = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    ).cuda()

    # CIFAR-10 dataset with enhanced augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.encoder_q.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    # Fine-tuning loop
    for epoch in range(100):  # Increased epochs to 100
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model.encoder_q(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # Save the encoder_q
    torch.save(model.encoder_q.state_dict(), "fine_tuned_encoder_q_cifar10_v2.pth")
    print("Fine-tuning complete. Model saved as 'fine_tuned_encoder_q_cifar10_v2.pth'.")


def evaluate_fine_tuned_model():
    print("Starting evaluation...")
    # Load the fine-tuned encoder_q
    model = FineTunedMoCo().cuda()
    model.encoder_q.load_state_dict(torch.load('fine_tuned_encoder_q_cifar10.pth'))
    print("Fine-tuned encoder_q loaded successfully.")

    model.eval()

    # CIFAR-10 test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Evaluate accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Fine-Tuned MoCo Accuracy on CIFAR-10: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    print("Fine-tuning MoCo with labeled CIFAR-10 data...")
    fine_tune_with_labels()  # Fine-tune the pretrained model
    print("Fine-tuning complete. Evaluating model...")
    accuracy = evaluate_fine_tuned_model()  # Evaluate the fine-tuned model
    print(f"Evaluation complete. Final Accuracy: {accuracy:.2f}%")
