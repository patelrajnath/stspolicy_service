import io
from configparser import ConfigParser
from pathlib import Path

from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import pandas as pd

# MODEL_DIR = os.environ["MODEL_DIR"]
# MODEL_FILE = os.environ["MODEL_FILE"]
# METADATA_FILE = os.environ["METADATA_FILE"]
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
# METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

# print("Loading model from: {}".format(MODEL_PATH))
# clf = load(MODEL_PATH)
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel

app = Flask(__name__)
api = Api(app)

import flask
flask.__version__

if __name__ == '__main__':
    train_args = {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,
        'max_seq_length': 80,
        'num_train_epochs': 1,
        'evaluate_during_training_steps': 100,
        'train_batch_size': 32,
        "output_dir": "outputs/",
        "best_model_dir": "outputs/best_model/",
        'regression': True,
        'use_multiprocessing': False,
        "wandb_project": False,
        'save_steps': 0,  # new from code inspect
        "gradient_accumulation_steps": 1,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False
    }

    model = ClassificationModel('roberta',
                                'roberta-base', num_labels=1, args=train_args,
                                use_cuda=False,
                                cuda_device=-1
                                )


    class CustomSpacyNER(Resource):
        def __init__(self) -> None:
            self._required_features_optional = ['intents', 'params']
            self.reqparse = reqparse.RequestParser()
            self.reqparse.add_argument(
                'text', type=list, required=True, location='json',
                help='No {} provided'.format('text'))
            for feature in self._required_features_optional:
                self.reqparse.add_argument(
                    feature, type=list, required=False, location='json',
                    help='No {} provided'.format(feature))
            super(CustomSpacyNER, self).__init__()

        def put(self):
            sts_df = pd.read_csv('data/STS-B/train.tsv', sep='\t', error_bad_lines=False)
            eval_df = pd.read_csv('data/STS-B/dev.tsv', sep='\t', error_bad_lines=False)

            sts_df = sts_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
            eval_df = eval_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()

            args = self.reqparse.parse_args()
            texts = args["text"]
            intents = args["intents"]

            columns = ['text_a', 'text_b']
            rows = []
            for t, i in zip(texts, intents):
                rows.append([t, i.replace('_', ' ')])

            le = LabelEncoder()
            nlu_df = pd.DataFrame(rows, columns=columns)
            nlu_df['le'] = le.fit_transform(nlu_df['text_b'])

            intent_map = dict()
            for idx, row in nlu_df.iterrows():
                label = row['le']
                intent_map[label] = row['text_b']

            num_intents = len(intent_map) - 1

            def get_random_label(intent_label):
                import random
                # random.seed(123)
                while True:
                    random_label = random.randint(0, num_intents)
                    if random_label != intent_label:
                        return random_label
                    continue

            def get_random_intent(intent_label):
                random_label = get_random_label(intent_label)
                return intent_map[random_label]

            num_rows, num_columns = nlu_df.shape
            values = [5] * (num_rows)
            nlu_df = nlu_df.assign(score=values)

            for idx, row in nlu_df.iterrows():
                label = row['le']
                random_intent = get_random_intent(label)
                nlu_df = nlu_df.append(nlu_df.loc[[idx] * 1].assign(score=0, text_b=random_intent), ignore_index=True)
            nlu_df = nlu_df.drop(columns=['le'])
            nlu_df = nlu_df.rename(columns={'score': 'labels'}).dropna()

            select_columns = ['text_a', 'text_b', 'labels']
            nlu_df = nlu_df[select_columns]
            sts_df = sts_df[select_columns]
            eval_df = eval_df[select_columns]

            train_df = pd.concat([nlu_df, sts_df])

            from scipy.stats import pearsonr, spearmanr

            def pearson_corr(preds, labels):
                return pearsonr(preds, labels)[0]

            def spearman_corr(preds, labels):
                return spearmanr(preds, labels)[0]

            global model
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr)

            result, model_outputs, wrong_predictions = \
                model.eval_model(eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr)
            print(result)
            return jsonify({'response': f' STSFallbackTrainer completed training!'})

        def get(self):
            global model
            print(model)
            args = self.reqparse.parse_args()
            for text in args["text"]:
                print(text)
            return {'entities': args["text"]}

    api.add_resource(CustomSpacyNER, '/train', '/predict')
    app.run(debug=False, host='0.0.0.0', port=9502)
