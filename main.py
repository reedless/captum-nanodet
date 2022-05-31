import os
import pickle
import warnings

from average_attrs import average_cosine_similarity

warnings.filterwarnings("ignore")

with open('split_imgs.pkl', 'rb') as f:
    split_imgs = pickle.load(f)

for epoch in [300, 500]:
    weights_path = f'assets/nanodet-{epoch}/nanodet-{epoch}.pth'
    config_path = f'assets/nanodet-{epoch}/nanodet-{epoch}.pth_train_config.yml'
    
    average_scores = []
    count = 0

    for key in sorted(os.listdir('dataset')):
        for i in range(len(split_imgs[key]["test"])):
            if count >= 5:
                break
            image_path = f'dataset/{key}/{split_imgs[key]["test"][i]}'
            try:
                average_scores.append(average_cosine_similarity(image_path, weights_path, config_path))
                count += 1
            except ValueError:
                continue

    print(f'Average cosine similarity for {epoch} epochs: {sum(average_scores)/len(average_scores)}')
