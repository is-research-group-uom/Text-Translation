from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
import evaluate
import nltk

def eval_dictionary(dict, from_language, to_language):
    rouge = evaluate.load('rouge')
    for data in dict:
        text = data[from_language]
        reference = data[to_language]
        response = data['ai_response']
        print("For the First sentence: ", text, "\nThe Correct sentence: ", reference, "\nAi generated Translation: ", response)

        # BERT score
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([response], [reference])
        print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

        # nltk.download('wordnet')
        # METEOR score
        score = meteor_score([reference.split()], response.split())
        print(f"METEOR Score: {score}")

        results = rouge.compute(predictions=[response], references=[reference])
        print(f"Rouge Score: {results}")
        print("---------------------------------------------")