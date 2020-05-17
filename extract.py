# import nltk
import os
import sys

# document = 'a_little_bit_of_time'

# fixed = ' '.join(document.split('_'))
# print(nltk.word_tokenize(fixed))


def main():
    #  Read data from files
    if len(sys.argv) < 2 :
        sys.exit("Usage: python extract.py directory [filename]")

    # create_files(sys.argv[1], sys.argv[2])
    create_files(sys.argv[1])


def format_words(words):
    if words.count('_') > 0:
        fixed = ' '.join(words.split('_'))
        return fixed
    return words

def extract_data(line):
    columns = line.split()


    words = columns[0].strip()
    
    words = format_words(words)

    polarity = columns[1].strip()
    intensity = columns[2].strip()

    return (polarity, [words, intensity])
    
    # results[polarity].append([words, intensity])
    # for i in range(len(columns)):
        

    #     if columns[1].strip()
    #     if i == 0:

    #         results[''].append(format_words(row.strip()))
    #     else:
    #         results.append(row.strip()) 


    # print(results)
    # return results

def load_data(filename):
    results = {}
    results['positive'] = []
    results['negative'] = []
    # with open(os.path.join(directory, filename)) as f:
    with open(filename) as f:
        for line in f.read().splitlines():
            # result.append(extract_data(line))
            polarity, data  = extract_data(line)
    
            results[polarity].append(data)

    return results


def load_dict():
    from senticnet4 import senticnet
    # 0,1,2,3,7
    results = {}
    results['positive'] = []
    results['negative'] = []
    for word in senticnet:
        selection = senticnet[word]
        polarity = selection[6]
        #       word, pleasantness value, attention value, sensitivity value, aptitude value, polarity value
        data = [format_words(word), selection[0], selection[1], selection[2], selection[3], selection[7]]
        results[polarity].append(data)
    return results

def create_files(directory, source=''):
    # data = load_data(source)
    data = load_dict()
    positives = data['positive']
    negatives = data['negative']

    with open(os.path.join(directory, 'positives.txt'), 'w') as f:
        for line in positives:
            s = ','.join(line)
            f.write(s + '\n')
    
    with open(os.path.join(directory, 'negatives.txt'), 'w') as f:
        for line in negatives:
            s = ','.join(line)
            f.write(s + '\n')


# create_files('small.txt', 'corpus1')

# print(load_data('small.txt')['negative'])

if __name__ == "__main__":
    main()
