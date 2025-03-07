import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from PneumoniaCNN import PneumoniaCNN
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    train_transforms = transforms.Compose([ # with data augmentation for training
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(root='data/chest_xray/chest_xray/train', transform=train_transforms)

    val_dataset = ImageFolder(root='data/chest_xray/chest_xray/val',   transform=test_transforms)

    test_dataset = ImageFolder(root='data/chest_xray/chest_xray/test',  transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    n_train_dataset = len(train_dataset)
    n_normal_train = sum([1 for _, label in train_dataset if label == 0])
    n_pneumonia_train = sum([1 for _, label in train_dataset if label == 1])
    n_classes = len(train_dataset.classes)

    w_normal = n_train_dataset / (n_classes * n_normal_train) *2 # we multiply by 2 because the normal class is under-represented
    w_pneumonia = n_train_dataset / (n_classes * n_pneumonia_train)
    weights = torch.tensor([w_normal, w_pneumonia]).to(device)
    print(f"Weight normal : {w_normal:.4f} | Weight pneumonia : {w_pneumonia:.4f}")

    print("Classes: ", n_classes)
    print("Train dataset size: ", n_train_dataset)
    print("Validation dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))

    # training part :

    model = PneumoniaCNN().to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # early stopping
    best_val_loss = float('inf')
    patience = 7
    counter = 0
    epochs_done = 0

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0) # total loss for all images in the batch
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_losses.append(running_loss/total)
        train_accs.append(correct/total)

        # validation part :

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)
                
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, labels_val)
                
                val_loss += loss_val.item() * images_val.size(0)
                _, predicted_val = torch.max(outputs_val, 1)
                val_correct += (predicted_val == labels_val).sum().item()
                val_total += labels_val.size(0)
        
        val_losses.append(val_loss/val_total)
        val_accs.append(val_correct/val_total)

        print(f"Epoch [{epoch+1}/{num_epochs}] \n",
            f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} \n",
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accs[-1]:.4f}")

        epochs_done += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'PneumoniaCNN.pth')
            print("Best model saved")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load('PneumoniaCNN.pth'))
    print("Best model loaded")
    # test part :

    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for images_test, labels_test in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)
            
            outputs_test = model(images_test)
            _, preds_test = torch.max(outputs_test, 1)
            
            all_labels.extend(labels_test.cpu().numpy())
            all_preds.extend(preds_test.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=train_dataset.classes,
               yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_CNN.png')

    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # plot the losses and the accuracies

    # plt.figure(figsize=(12, 5))
    
    # # loss subplot
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs_done+1), train_losses, 'b-', label='Training Loss', marker='x')
    # plt.plot(range(1, epochs_done+1), val_losses, 'r-', label='Validation Loss', marker='x')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True)
    
    # # Accuracy subplot
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs_done+1), train_accs, 'b-', label='Training Accuracy', marker='x')
    # plt.plot(range(1, epochs_done+1), val_accs, 'r-', label='Validation Accuracy', marker='x')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('training_curves_CNN.png')