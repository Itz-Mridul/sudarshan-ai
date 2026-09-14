from src.data_pipeline.dataset import get_dataloaders
if __name__ == '__main__':
    train_l, val_l, test_l = get_dataloaders("data/clean", "data/stego", max_images=10, num_workers=0)
    for c, s in val_l:
        print(c.shape, s.shape)
        break
