import os
import numpy as np
import pandas as pd
from transformers import pipeline
from transformers.image_utils import load_image
from util import pu_learning,one_class_svm,iforest_learning,lof_learning

local_model_path = "../dinov3-vitl16-pretrain-lvd1689m" 

feature_extractor = pipeline(
    model=local_model_path,         
    task="image-feature-extraction", 
    device="cuda"                    
)

def extract_features(image_path):
    features = []
    names = []
    files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Extracting features from {image_path}...")
    
    for f in files:
        image = load_image(os.path.join(image_path, f))
        output = feature_extractor(image)
        features.append(output[0][0])
        names.append(f)

    return np.array(features), names


if __name__ == '__main__':
    train_dir = "../../train/images"
    test_dir = "../../test/images"
    out_put = './submission.csv'
    train_feats, train_names = extract_features(train_dir)
    test_feats, test_names = extract_features(test_dir)

    pu_label = pu_learning(train_feats, test_feats)
    ocs = one_class_svm(train_feats, test_feats)
    iforest = iforest_learning(train_feats, test_feats)
    lof = lof_learning(train_feats, test_feats)
    final_labels = []
    for i in range(len(test_names)):
        scores = [0,0]
        scores[pu_label[i]] += 1.2
        scores[ocs[i]] += 0.5
        scores[iforest[i]] += 0.5
        scores[lof[i]] += 0.5
        final_label = np.argmax(scores)
        final_labels.append(final_label)

    df = pd.DataFrame({"id": test_names, "label": final_labels}) 
    df['sort_id'] = df['id'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    df.sort_values(by='sort_id', inplace=True)
    df.drop(columns=['sort_id'], inplace=True)
    df.to_csv(out_put, index=False)
