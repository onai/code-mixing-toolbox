from fastText import load_model
import joblib
import numpy as np
import sys

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def is_fence_word(w_embed, center1, center2):
    distance1 = np.linalg.norm(w_embed - centers[cluster1])
    distance2 = np.linalg.norm(w_embed - centers[cluster2])
            
    the_dist = abs(distance1 - distance2) / (np.linalg.norm(centers[cluster1] - centers[cluster2]))

    return the_dist < 0.1
    
if __name__ == '__main__':
    language_model_f = sys.argv[1]
    cluster_model_f = sys.argv[2]
    sentences = sys.argv[3]
    cluster1 = int(sys.argv[4])
    cluster2 = int(sys.argv[5])
    model = load_model(language_model_f)
    kmeans = joblib.load(cluster_model_f)
    centers = kmeans.cluster_centers_

    sentence_dist = {}
    
    with open(sentences) as handle:
        for new_line in handle:

            if len(new_line.split()) < 10:
                continue

            new_line = new_line.strip()
            #if not isEnglish(new_line):
            #    continue

            if len(set(new_line.split())) < 10:
                continue

            words = new_line.split()

            v = model.get_sentence_vector(new_line)
            if kmeans.predict([v])[0] != cluster1 and  kmeans.predict([v])[0] != cluster2:
                continue
            

            word_embeds = [model.get_word_vector(w) for w in words]

            word_embeds_normed = [w / np.linalg.norm(w) for w in word_embeds]

            fence_words = []
            non_fence_words = []
            nf_embeds = []

            for i, w in enumerate(words):
                embed = word_embeds[i]

                if is_fence_word(embed, kmeans.cluster_centers_[cluster1], kmeans.cluster_centers_[cluster2]):
                    fence_words.append(w)
                else:
                    non_fence_words.append(w)
                    nf_embeds.append(word_embeds[i])

            u = len(fence_words)
            n_f = len(non_fence_words)

            if len(nf_embeds) == 0:
                print('ZERO_DENOM', new_line.strip())
                continue

            word_preds = kmeans.predict(nf_embeds)

            lang_freqs = {}

            for pred in word_preds:
                if pred in lang_freqs:
                    lang_freqs[pred] += 1
                else:
                    lang_freqs[pred] = 1

            max_lang = 0
            for k, v in lang_freqs.items():
                if v > max_lang:
                    max_lang = v

            max_wi = max_lang

            n = len(words)

            if n - u == 0:
                print('ZERO_DENOM', new_line.strip())
                continue
            
            cmi = (n_f - max_wi) / float(n - u)
            
            # distance1 = np.linalg.norm(v - centers[cluster1])
            # distance2 = np.linalg.norm(v - centers[cluster2])
            
            # the_dist = abs(distance1 - distance2) / (np.linalg.norm(centers[cluster1] - centers[cluster2]))

            print(cmi, new_line)
