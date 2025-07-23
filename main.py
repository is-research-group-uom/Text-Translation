from llm.claude3_5 import claude3_5
from datasets import load_dataset
from evaluations import eval_dictionary
from llm.llama_30b import llama
from llm.deepseek_r1 import deepseek


print("Give a number for the following translations:")
print("1. el-en\n2. en-it\n3. de-en\n4. en-el\n5. it-en\n6. en-de")

translations = {
    '1': ('el-en', 'el', 'en'),
    '2': ('en-it', 'en', 'it'),
    '3': ('de-en', 'de', 'en'),
    '4': ('el-en', 'en', 'el'),
    '5': ('en-it', 'it', 'en'),
    '6': ('de-en', 'en', 'de')
}

x = input()
if x in translations:
    translation, from_language, to_language = translations[x]
    print(f"Translation: {translation}, from Language: {from_language},To Language: {to_language}")
else:
    print("Invalid choice")


ds = load_dataset("Helsinki-NLP/europarl", name=translation)
print(ds)
train_data = ds["train"]

dict_eval = []
for i in range(100):
    comment = train_data[i]
    text = comment["translation"][from_language]
    response = llama(text, from_language, to_language)
    print(response,"\n---------------------------------------")

    dict = {
        from_language: text,
        to_language: comment['translation'][to_language],
        'ai_response': response
    }

    dict_eval.append(dict)

eval_dictionary(dict_eval, from_language, to_language)