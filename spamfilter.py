import pandas as pd
import plotly.express as px
from dplython import DplyFrame

# define global variables
data_set = None
train_set = None
test_set = None

word_avg = dict()

SPAM = 0
NONSPAM = 1

alpha = .0001

words = ['make', 'address', 'all', 'num3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', 'num000', 'money', 'hp', 'hpl', 'george', 'num650', 'lab', 'labs', 'telnet', 'num857', 'data', 'num415', 'num85', 'technology', 'num1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', 'charSemicolon', 'charRoundbracket', 'charSquarebracket', 'charExclamation', 'charDollar', 'charHash']


def read_in_data():
    """ Read in the data from the Spam.xlsx """
    # get the global var
    global data_set

    # read in the data from the excel file
    data_set = pd.read_excel('Spam.xlsx')
    data_set = DplyFrame(data_set)



def split_data_into_sets():
    """ Split the read in data into one smaller and one bigger set """
    # get the global vars
    global train_set
    global test_set

    # read in the data from the excel file
    train_set = data_set.sample(frac=0.8, random_state=1)
    test_set = data_set.sample(frac=0.2, random_state=2)


def calculate_pes(word) -> float:
    """Calculate the value of P(E|S)

    :param word: String with a word from the xlsx header
    :returns: The Calculated P(E|S) as Float
    """
    # get the global train set
    global train_set
    global word_cuts

    # calculate the e -> count(s / e)
    e = train_set.type[(train_set.type == 'spam') & (train_set[word] >= word_cuts[word][1])].count()
    # calculate the s -> count(s)
    s = train_set.type[(train_set.type == 'spam')].count()

    # cast values to floats else it will print 0
    return float(e) + alpha / float(s)




def calculate_pens(word) -> float:
    """Calculates the P(E|Sc)

    :param message: String with a word from the xlsx header
    :returns: The Calculated P(E|S) as Float
    """

    global train_set
    global word_cuts

    # calculate the e -> count(s / e)
    e = train_set.type[(train_set.type == 'nonspam') & (train_set[word] >= word_cuts[word][1])].count()
    # calculate the s -> count(s)
    s = train_set.type[(train_set.type == 'nonspam')].count()

    # cast values to floats el
    return float(e) + alpha / float(s)




def calculate_pesc(word):
    """Calculates the P(E|Sc)

    :param message: String with a word from the xlsx header
    :returns: The Calculated P(E|S) as Float
    """

    e = train_set >> sift((X[word] >= 3) & (X.type != "spam")) >> summarize(n=X.type.count())
    s = train_set >> sift((X.type != 'spam')) >> summarize(word=X.type.count())
    p = e / s
    print(p)


def calculate_pse(word):
    """Calculates the P(S|E)

    :param message: String with a word from the xlsx header
    :returns: The Calculated P(S|E) as Float
    """

    s = train_set >> sift((X[word] >= 3)) >> summarize(n=X.type.count())
    e = train_set >> sift((X[word] >= 3) & (X.type == 'spam')) >> summarize(n=X.type.count())

    p = e.n / s.n
    print(p)


def get_measures(data_set):
    """Get the measures for the existing Dataset.

    :param data_set: The existing data
    :returns: This function returns the measures for the data_set
    """

    mat = pd.crosstab(get_prediction(data_set)['prediction'], data_set.type, margins=True)

    measures = pd.DataFrame({'Err': (float(mat.iloc[0, 1]) + float(mat.iloc[1, 0])) / float(mat.iloc[2, 2]),
                             'Acc': 1.0 - (float(mat.iloc[0, 1]) + float(mat.iloc[1, 0])) / float(mat.iloc[2, 2]),
                             'Sens': float(mat.iloc[1, 1])/float(mat.iloc[2, 1]),
                             'Spez': float(mat.iloc[0, 0]) / float(mat.iloc[2, 0]),
                             'PV+': float(mat.iloc[1, 1]) / float(mat.iloc[1, 2]),
                             'PV-': float(mat.iloc[0, 0]) / float(mat.iloc[0, 2])}, index=[0])
    return measures


def get_measures_for_word(data_set, word, cut_value):
    """Get the measures for the existing Dataset, word and cut_value.

    :param data_set: The existing data
    :param word: A string with a word inside from the xlsx
    :param cut_value: The cut_value of the existing
    :returns: This Function will return the calculated measures for the word
    """

    # Create a crosstab from the word prediction
    mat = pd.crosstab(get_prediction_for_word(data_set, word, cut_value)['prediction'], data_set.type, margins=True)

    measures = pd.DataFrame({'Err': (float(mat.iloc[0, 1]) + float(mat.iloc[1, 0])) / float(mat.iloc[2, 2]),
                             'Acc': 1.0 - (float(mat.iloc[0, 1]) + float(mat.iloc[1, 0])) / float(mat.iloc[2, 2]),
                             'Sens': float(mat.iloc[1, 1])/float(mat.iloc[2, 1]),
                             'Spez': float(mat.iloc[0, 0]) / float(mat.iloc[2, 0]),
                             'PV+': float(mat.iloc[1, 1]) / float(mat.iloc[1, 2]),
                             'PV-': float(mat.iloc[0, 0]) / float(mat.iloc[0, 2])}, index=[0])
    return measures


def get_prediction(data_set):
    """Get the prediction for the data_set

    :param data_set: The existing data
    """

    data_set['prediction'] = 'pred_ham'
    cs = 0
    cns = 0
    for i, row in data_set.iterrows():
        spam = 1
        nspam = 1
        for w in words:
            spam *= (row[w] * word_avg[w][SPAM] + alpha)
            nspam *= (row[w] * word_avg[w][NONSPAM] + alpha)

        spam *= Pspam
        nspam *= Nspam

        res = spam / (spam + nspam) > .88
        if res:
            cs += 1
            data_set.loc[i, 'prediction'] = 'pred_spam'
        else:
            cns += 1

    print(cs)
    print(cns)

    return data_set


def get_prediction_for_word(data_set, word, cut_value):
    """Get the prediction for the existing Word.

    :param word: A String with the name of the row
    :param data_set: The existing data
    :param cut_value: The cut_value of the word
    :returns: The updated data_set
    """

    data_set['prediction'] = 'pred_ham'
    data_set.loc[data_set[word] >= cut_value, 'prediction'] = 'pred_spam'
    return data_set


def cut_values(word):
    global word_cuts

    keep_count = pd.DataFrame(None)
    count = 0
    for i in range(1, 60):
        cut_value = 0.001 + count/100.0
        temp_data = get_prediction_for_word(train_set, word, cut_value)
        new = get_measures_for_word(temp_data, word, cut_value)

        if (count == 0): df = new
        if (count != 0): df = df.append(new)

        keep_count.at[count, 'cut_value'] = cut_value
        count = count + 1

    df.index = list(range(len(df.index)))
    df['cut_value'] = keep_count

    word_cuts[word] = get_max_cut(df)


def get_max_cut(df):
    """Get the measures for the existing Dataset.

    :param df: The existing data
    """

    max = 0
    cut = 0

    for i, row in df.iterrows():
        if row['Acc'] > max:
            max = row['Acc']
            cut = row['cut_value']

    return [cut, max]


def classify(data_set):
    """Classify if spam or ham

    :param data_set: Existing data_set
    """

    print(word_cuts)
    cc = 0
    cfs = 0
    cfn = 0
    for i, row in data_set.iterrows():
        is_spam = False
        spamc = 0
        for t in word_cuts:
            if t[2] < 0.65:
                continue
            if row[t[0]] > t[1]:
                spamc += 1
            else:
                spamc -= 1
        if spamc >= 0:
            is_spam = True

        if is_spam and row['type'] == 'spam':
            cc += 1
        elif is_spam and row['type'] == 'nonspam':
            cfs += 1
        elif not is_spam and row['type'] == 'spam':
            cfn += 1
        else:
            cc += 1

    print(f"Correct: {cc}")
    print(f"False spam: {cfs}")
    print(f"False nonspam: {cfn}")


def calc_avg(data_set, word):
    """Calculate the average of a given word

    :param data_set: Existing data_set
    :param word: A String with the name of the row
    """
    res = []

    res.append(data_set[word][data_set.type == 'spam'].sum() / data_set.type[data_set.type == 'spam'].count())

    res.append(data_set[word][data_set.type == 'nonspam'].sum() / data_set.type[data_set.type == 'nonspam'].count())

    word_avg[word] = res


read_in_data()
split_data_into_sets()

Pspam = train_set.type[(train_set.type == 'spam')].count() / train_set.type.count()
Nspam = train_set.type[(train_set.type == 'nonspam')].count() / train_set.type.count()

for w in words:
    calc_avg(train_set, w)

print(get_measures(test_set))


