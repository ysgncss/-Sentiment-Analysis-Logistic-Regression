import argparse
import logging
import math
import random
import re

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fixing random seed for reproducibility
random.seed(123)
np.random.seed(123)

stop_words = ['a', 'in', 'on', 'at', 'and', 'or',
              'to', 'the', 'of', 'an', 'by',
              'as', 'is', 'was', 'were', 'been', 'be',
              'are', 'for', 'this', 'that', 'these', 'those', 'you', 'i',
              'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'his', 'her', 'they', 'them', 'from', 'with', 'its']


def extract_ngrams(x_raw, ngram_range=(1, 3), token_pattern=r'',
                   stop_words=[], vocab=set(), char_ngrams=False):
    if char_ngrams:
        _white_spaces = re.compile(r"\s\s+")
        text_document = _white_spaces.sub(" ", x_raw)
        text_len = len(text_document)

        min_n, max_n = ngram_range
        if min_n == 1:
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        ngrams_append = ngrams.append
        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams

    else:
        # handle stop words
        x_raw = x_raw.split(" ")
        if stop_words is not None:
            tokens = [w for w in x_raw if w not in stop_words]

        min_n, max_n = ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            tokens_append = tokens.append
            space_join = " ".join
            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))
        return tokens


def get_vocab(X_raw, ngram_range=(1, 3), token_pattern=r'',
              min_df=0, keep_topN=0,
              stop_words=[], char_ngrams=False):
    vocab_dic = {}
    for x_raw in X_raw:
        features = extract_ngrams(
            x_raw, ngram_range=ngram_range, token_pattern=token_pattern,
            stop_words=stop_words, vocab=set(), char_ngrams=char_ngrams
        )

        for feature in features:
            vocab_dic[feature] = vocab_dic.get(feature, 0) + 1

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_df], key=lambda x: x[1], reverse=True)
    if keep_topN > 0:
        vocab_list = vocab_list[: keep_topN]

    vocab_dic = {word_count[0]: word_count[1] for idx, word_count in enumerate(vocab_list)}
    vocab, df, ngram_counts = list(vocab_dic.keys()), vocab_dic, list(vocab_dic.values())
    return vocab, df, ngram_counts


def get_vocab_mixed(X_raw, ngram_range=(1, 3), token_pattern=r'',
                    min_df=0, keep_topN=0,
                    stop_words=[]):
    vocab_dic = {}
    for x_raw in X_raw:
        char_features = extract_ngrams(
            x_raw, ngram_range=ngram_range, token_pattern=token_pattern,
            stop_words=stop_words, vocab=set(), char_ngrams=True
        )
        word_features = extract_ngrams(
            x_raw, ngram_range=ngram_range, token_pattern=token_pattern,
            stop_words=stop_words, vocab=set(), char_ngrams=False
        )

        features = char_features + word_features
        for feature in features:
            vocab_dic[feature] = vocab_dic.get(feature, 0) + 1

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_df], key=lambda x: x[1], reverse=True)
    if keep_topN > 0:
        vocab_list = vocab_list[: keep_topN]

    vocab_dic = {word_count[0]: word_count[1] for idx, word_count in enumerate(vocab_list)}
    vocab, df, ngram_counts = list(vocab_dic.keys()), vocab_dic, list(vocab_dic.values())
    return vocab, df, ngram_counts


def vectorise(X_ngram, vocab):
    v_to_idx = {}
    for v in vocab:
        v_to_idx[v] = len(v_to_idx)

    X_ngram_vec = []
    for x_ngram in X_ngram:
        x_vec = [0] * len(vocab)
        for item_ngram in x_ngram:
            if item_ngram in v_to_idx:
                x_vec[v_to_idx[item_ngram]] = x_vec[v_to_idx[item_ngram]] + 1

        X_ngram_vec.append(x_vec)
    return np.array(X_ngram_vec)

# get how many files each word appears in
def get_ngram_file_count(text_ngram):
    cotains_ngram_file_counts = {}
    for item_text in text_ngram:
        dup_item_text = set(item_text)
        for ngram in dup_item_text:
            cotains_ngram_file_counts[ngram] = cotains_ngram_file_counts.get(ngram, 0) + 1
    return cotains_ngram_file_counts


# compute idf
def idf(df, text_ngram, cotains_ngram_file_counts):
    idf_dic = {}
    for v in df.keys():
        # IDF：log(D/(counts(v)+1))
        idf = 1 + math.log((len(text_ngram) + 1) / (cotains_ngram_file_counts[v] + 1))
        idf_dic[v] = idf
    return idf_dic


# compute tf_idf
# def tf_idf(df, idf_dic):
#     print(df)
#     print(idf_dic)
#     tf_idf_dic = {}
#     total_grams = sum(df.values())
#     for v in df.keys():
#         # IDF：log(D/(counts(v)+1))
#         tf = df[v] / total_grams
#         tf_idf_dic[v] = tf * idf_dic[v]
#     return tf_idf_dic


def transform_count_tf_idf(count_vec, idf_dic, vocab):
    v_to_idx = {}
    for v in vocab:
        v_to_idx[v] = len(v_to_idx)
    idx_to_v = {v: k for k, v in v_to_idx.items()}

    X_tf_idf_ngram_vec = []
    for item in count_vec:
        x_tf_idf_vec = [0] * len(item)
        for idx in range(0, len(item)):
            if item[idx] != 0:
                # compute tf * idf
                x_tf_idf_vec[idx] = item[idx] * idf_dic[idx_to_v[idx]]

        X_tf_idf_ngram_vec.append(x_tf_idf_vec)
    return np.array(X_tf_idf_ngram_vec)


# def predict_proba(X, weights):
#     prob = sigmoid(dot(X, weights))
#     return prob


def predict_class(X, weights):
    y_pred = np.round(predict_proba(X, weights))
    return y_pred


def binary_loss(X, Y, weights, alpha=0.00001):
    '''
    Binary Cross-entropy Loss

    X:(len(X),len(vocab))
    Y: array len(Y)
    weights: array len(X)
    '''

    Y_pred = predict_proba(X, weights)
    loss = - Y * np.log(Y_pred) - (1 - Y) * np.log(1 - Y_pred) + alpha * np.sum(weights ** 2)
    loss = np.mean(loss)
    return loss


# 计算输入的sigmoid
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


# 在给定权重和偏差的情况下，找出模型预测输出1的概率
def predict_proba(X, w):
    return sigmoid(X.dot(w))


# 打乱一列原来的顺序
def shuffle(X, Y):
    randomsize = np.arange(len(X))
    np.random.shuffle(randomsize)
    return (X[randomsize], Y[randomsize])


# 通过数学推导损失函数,求各个参数的梯度
def get_gradient_regularization(X, Y, w, lamda):
    y_pred = predict_proba(X, w)
    pred_error = Y - y_pred
    w_grad = -np.sum(np.multiply(pred_error.T, X.T), 1) + lamda * w
    return w_grad


def get_accuracy(Y_pred, Y_label):
    Y_label = list(map(int, Y_label))
    acc = np.sum(Y_pred == Y_label) / len(Y_label)
    return acc


def SGD(X_tr, Y_tr, X_dev=[], Y_dev=[], lr=0.1,
        alpha=0.00001, epochs=5,
        tolerance=0.0001, print_progress=True):
    weights =  np.random.normal(size=X_tr[0].shape)
    training_loss_history = []
    training_acc_history = []
    validation_loss_history = []
    validation_acc_history = []

    batch_size = 16
    step = 0

    pre_validation_loss = np.inf
    for i in range(1, epochs + 1):
        # 打乱每次训练的数据
        X_train, Y_train = shuffle(X_tr, Y_tr)

        # 逻辑回归按批次训练
        for idx in range(int(np.floor(len(Y_train) / batch_size))):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
            # 计算梯度损失
            w_grad = get_gradient_regularization(X, Y, weights,  alpha)
            # Y = list(map(int, Y))
            # 梯度更新
            weights -= lr / np.sqrt(step) * w_grad

            step += 1



        # 在每个epoch训练中记录下训练误差 以及验证集中的误差用于画图数据
        y_train_pred = predict_proba(X_train, weights)
        Y_train_pred = np.round(y_train_pred)
        train_acc = get_accuracy(Y_train_pred, Y_train)
        training_acc_history.append(train_acc)

        y_dev_pred = predict_proba(X_dev, weights)
        Y_dev_pred = np.round(y_dev_pred)
        validation_acc = get_accuracy(Y_dev_pred, Y_dev)
        validation_acc_history.append(validation_acc)

        # compute loss
        train_loss = binary_loss(X_tr, Y_tr, weights, alpha=alpha)
        validation_loss = binary_loss(X_dev, Y_dev, weights, alpha=alpha)
        training_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)

        if print_progress:
            print(
                "epoch: {}, train loss: {} acc:{}, dev loss:{} acc:{}".format(i, train_loss, train_acc, validation_loss,
                                                                              validation_acc))
            logging.log(
                level=logging.INFO,
                msg="epoch: {}, train loss: {} acc:{}, dev loss:{} acc:{}".format(i, train_loss, train_acc,
                                                                                  validation_loss,
                                                                                  validation_acc)
            )
        if math.fabs(pre_validation_loss - validation_loss) < tolerance:
            break

        pre_validation_loss = validation_loss

    return weights, training_loss_history, validation_loss_history


def Normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)

    res = []
    for item in data:
        new_list = []
        for j in item:
            new_list.append((float(j) - m) / (mx - mn))
        res.append(new_list)
    return np.array(res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--model', type=str, default='BOWBOCN-tfidf',
                        help='choose a model: BOW-count, BOW-tfidf, BOCN-count, BOCN-tfidf, BOWBOCN-count,BOWBOCN-tfidf')
    parser.add_argument('--ngram_range', type=tuple, default=(1, 3), help='the range of ngram')
    parser.add_argument('--min_df', default=10, type=int, help='ngram minimum frequency')
    parser.add_argument('--keep_topN', default=10000, type=int, help='maximum number of vocab')
    parser.add_argument('--stop_words', default=True, type=bool, help='remove stop words or not')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--alpha', default=0.0001, type=float, help='regularisation strength')
    parser.add_argument('--epochs', default=1000, type=int, help='number of full passes over the training data')
    parser.add_argument('--tolerance', default=1e-6, type=float,
                        help='stop training if the difference between the current and previous validation loss is smaller than a threshold')

    logging.basicConfig(filename='log', level=logging.INFO, format=" %(message)s")

    args = parser.parse_args()
    ngram_range = args.ngram_range
    model = args.model
    min_df = args.min_df
    keep_topN = args.keep_topN
    stop_words = stop_words if args.stop_words else []
    lr = args.lr
    alpha = args.alpha
    epochs = args.epochs
    tolerance = args.tolerance

    logging.log(level=logging.INFO,
                msg="ngram_range:{} model:{} min_df:{} keep_topN:{} stop_words:{} lr:{} alpha:{} epochs:{} tolerance:{}".format(
                    ngram_range, model, min_df, keep_topN, stop_words, lr, alpha, epochs, tolerance))

    train_dataset = pd.read_csv(
        './data_sentiment/train.csv',
        header=None, names=['text', 'label'],
        engine='python', encoding="utf8"
    )
    dev_dataset = pd.read_csv(
        './data_sentiment/dev.csv',
        header=None, names=['text', 'label'],
        engine='python', encoding="utf8"
    )
    test_dataset = pd.read_csv(
        './data_sentiment/test.csv',
        header=None, names=['text', 'label'],
        engine='python', encoding="utf8"
    )

    # print(train_dataset.sample(10))

    train_text = train_dataset['text'].values.tolist()
    train_label = np.array(train_dataset['label'].values)
    dev_text = dev_dataset['text'].values.tolist()
    dev_label = np.array(dev_dataset['label'].values)
    test_text = test_dataset['text'].values.tolist()
    test_label = np.array(test_dataset['label'].values)

    dataset_text = train_text + dev_text + test_text

    if model.startswith('BOW-'):
        vocab, df, ngram_counts = get_vocab(
            dataset_text,
            ngram_range=ngram_range,
            token_pattern=r'',
            min_df=args.min_df,
            keep_topN=keep_topN,
            stop_words=stop_words,
            char_ngrams=False
        )
        # vocab to id and id to vocab dic
        vocab_to_id = {}
        id_to_vocab = {}
        for k in df:
            vocab_to_id[k] = len(vocab_to_id)
        id_to_vocab = {v: k for k, v in vocab_to_id.items()}

        # get train ngram list by extract_ngrams()
        train_text_ngram = []
        for item in train_text:
            train_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            train_text_ngram.append(train_item_ngrams)

        # get dev ngram list by extract_ngrams()
        dev_text_ngram = []
        for item in dev_text:
            dev_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            dev_text_ngram.append(dev_item_ngrams)

        # get test ngram list by extract_ngrams()
        test_text_ngram = []
        for item in test_text:
            test_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            test_text_ngram.append(test_item_ngrams)

        # all text ngram list
        text_ngram = train_text_ngram + dev_text_ngram + test_text_ngram

        # get count_vec by vectorise()
        train_X_count_vec = vectorise(train_text_ngram, vocab)
        dev_X_count_vec = vectorise(dev_text_ngram, vocab)
        test_X_count_vec = vectorise(test_text_ngram, vocab)

        if model == 'BOW-count':
            train_X_count_vec = Normalize(train_X_count_vec)
            dev_X_count_vec = Normalize(dev_X_count_vec)
            test_X_count_vec = Normalize(test_X_count_vec)

            # training
            weights, training_loss_history, validation_loss_history = SGD(
                train_X_count_vec,
                train_label,
                X_dev=dev_X_count_vec,
                Y_dev=dev_label,
                lr=lr,
                alpha=alpha,
                epochs=epochs,
                tolerance=tolerance,
                print_progress=True
            )

            # testing
            preds_te_count = predict_class(test_X_count_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))
        elif model == 'BOW-tfidf':
            cotains_ngram_file_counts_dic = get_ngram_file_count(text_ngram)
            idf_dic = idf(df, text_ngram, cotains_ngram_file_counts_dic)

            # get tf.idf vectors
            train_X_tf_idf_vec = transform_count_tf_idf(train_X_count_vec, idf_dic, vocab)
            dev_X_tf_idf_vec = transform_count_tf_idf(dev_X_count_vec, idf_dic, vocab)
            test_X_tf_idf_vec = transform_count_tf_idf(test_X_count_vec, idf_dic, vocab)

            # Normalize
            train_X_tf_idf_vec = Normalize(train_X_tf_idf_vec)
            dev_X_tf_idf_vec = Normalize(dev_X_tf_idf_vec)
            test_X_tf_idf_vec = Normalize(test_X_tf_idf_vec)

            weights, training_loss_history, validation_loss_history = SGD(
                train_X_tf_idf_vec, train_label,
                X_dev=dev_X_tf_idf_vec,
                Y_dev=dev_label, lr=lr,
                alpha=alpha, epochs=epochs,
                tolerance=tolerance, print_progress=True
            )
            # testing
            preds_te_count = predict_class(test_X_tf_idf_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))
    elif model.startswith("BOCN-"):
        vocab, df, ngram_counts = get_vocab(
            dataset_text,
            ngram_range=ngram_range,
            token_pattern=r'',
            min_df=args.min_df,
            keep_topN=keep_topN,
            stop_words=stop_words,
            char_ngrams=True
        )
        # vocab to id and id to vocab dic
        vocab_to_id = {}
        id_to_vocab = {}
        for k in df:
            vocab_to_id[k] = len(vocab_to_id)
        id_to_vocab = {v: k for k, v in vocab_to_id.items()}

        # get train ngram list by extract_ngrams()
        train_text_ngram = []
        for item in train_text:
            train_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            train_text_ngram.append(train_item_ngrams)

        # get dev ngram list by extract_ngrams()
        dev_text_ngram = []
        for item in dev_text:
            dev_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            dev_text_ngram.append(dev_item_ngrams)

        # get test ngram list by extract_ngrams()
        test_text_ngram = []
        for item in test_text:
            test_item_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            test_text_ngram.append(test_item_ngrams)

        # all text ngram list
        text_ngram = train_text_ngram + dev_text_ngram + test_text_ngram

        # get count_vec by vectorise()
        train_X_count_vec = vectorise(train_text_ngram, vocab)
        dev_X_count_vec = vectorise(dev_text_ngram, vocab)
        test_X_count_vec = vectorise(test_text_ngram, vocab)


        if model == 'BOCN-count':

            train_X_count_vec = Normalize(train_X_count_vec)
            dev_X_count_vec = Normalize(dev_X_count_vec)
            test_X_count_vec = Normalize(test_X_count_vec)

            # training
            weights, training_loss_history, validation_loss_history = SGD(
                train_X_count_vec,
                train_label,
                X_dev=dev_X_count_vec,
                Y_dev=dev_label,
                lr=lr,
                alpha=alpha,
                epochs=epochs,
                tolerance=tolerance,
                print_progress=True
            )

            # testing
            preds_te_count = predict_class(test_X_count_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))
        elif model == 'BOCN-tfidf':
            cotains_ngram_file_counts_dic = get_ngram_file_count(text_ngram)
            idf_dic = idf(df, text_ngram, cotains_ngram_file_counts_dic)

            # get tf.idf vectors
            train_X_tf_idf_vec = transform_count_tf_idf(train_X_count_vec, idf_dic, vocab)
            dev_X_tf_idf_vec = transform_count_tf_idf(dev_X_count_vec, idf_dic, vocab)
            test_X_tf_idf_vec = transform_count_tf_idf(test_X_count_vec, idf_dic, vocab)

            # Normalize
            train_X_tf_idf_vec = Normalize(train_X_tf_idf_vec)
            dev_X_tf_idf_vec = Normalize(dev_X_tf_idf_vec)
            test_X_tf_idf_vec = Normalize(test_X_tf_idf_vec)

            weights, training_loss_history, validation_loss_history = SGD(
                train_X_tf_idf_vec, train_label,
                X_dev=dev_X_tf_idf_vec,
                Y_dev=dev_label, lr=lr,
                alpha=alpha, epochs=epochs,
                tolerance=tolerance, print_progress=True
            )
            # testing
            preds_te_count = predict_class(test_X_tf_idf_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))

    elif model.startswith("BOWBOCN-"):

        vocab, df, ngram_counts = get_vocab_mixed(
            dataset_text,
            ngram_range=ngram_range,
            token_pattern=r'',
            min_df=args.min_df,
            keep_topN=keep_topN,
            stop_words=stop_words
        )
        # vocab to id and id to vocab dic
        vocab_to_id = {}
        id_to_vocab = {}
        for k in df:
            vocab_to_id[k] = len(vocab_to_id)
        id_to_vocab = {v: k for k, v in vocab_to_id.items()}

        # get train ngram list by extract_ngrams()
        train_text_ngram = []
        for item in train_text:
            train_item_char_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            train_item_word_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            train_item_ngrams = train_item_char_ngrams + train_item_word_ngrams
            train_text_ngram.append(train_item_ngrams)

        # get dev ngram list by extract_ngrams()
        dev_text_ngram = []
        for item in dev_text:
            dev_item_char_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            dev_item_word_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            dev_item_ngrams = dev_item_char_ngrams + dev_item_word_ngrams
            dev_text_ngram.append(dev_item_ngrams)

        # get test ngram list by extract_ngrams()
        test_text_ngram = []
        for item in test_text:
            test_item_char_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=True
            )
            test_item_word_ngrams = extract_ngrams(
                item, ngram_range=ngram_range, token_pattern=r'', stop_words=stop_words,
                vocab=set(), char_ngrams=False
            )
            test_item_ngrams = test_item_char_ngrams + test_item_word_ngrams
            test_text_ngram.append(test_item_ngrams)

        # all text ngram list
        text_ngram = train_text_ngram + dev_text_ngram + test_text_ngram

        # get count_vec by vectorise()
        train_X_count_vec = vectorise(train_text_ngram, vocab)
        dev_X_count_vec = vectorise(dev_text_ngram, vocab)
        test_X_count_vec = vectorise(test_text_ngram, vocab)

        if model == 'BOWBOCN-count':

            train_X_count_vec = Normalize(train_X_count_vec)
            dev_X_count_vec = Normalize(dev_X_count_vec)
            test_X_count_vec = Normalize(test_X_count_vec)

            # training
            weights, training_loss_history, validation_loss_history = SGD(
                train_X_count_vec,
                train_label,
                X_dev=dev_X_count_vec,
                Y_dev=dev_label,
                lr=lr,
                alpha=alpha,
                epochs=epochs,
                tolerance=tolerance,
                print_progress=True
            )

            # testing
            preds_te_count = predict_class(test_X_count_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))
        elif model == 'BOWBOCN-tfidf':
            cotains_ngram_file_counts_dic = get_ngram_file_count(text_ngram)
            idf_dic = idf(df, text_ngram, cotains_ngram_file_counts_dic)

            # get tf.idf vectors
            train_X_tf_idf_vec = transform_count_tf_idf(train_X_count_vec, idf_dic, vocab)
            dev_X_tf_idf_vec = transform_count_tf_idf(dev_X_count_vec, idf_dic, vocab)
            test_X_tf_idf_vec = transform_count_tf_idf(test_X_count_vec, idf_dic, vocab)

            # Normalize
            train_X_tf_idf_vec = Normalize(train_X_tf_idf_vec)
            dev_X_tf_idf_vec = Normalize(dev_X_tf_idf_vec)
            test_X_tf_idf_vec = Normalize(test_X_tf_idf_vec)

            weights, training_loss_history, validation_loss_history = SGD(
                train_X_tf_idf_vec, train_label,
                X_dev=dev_X_tf_idf_vec,
                Y_dev=dev_label, lr=lr,
                alpha=alpha, epochs=epochs,
                tolerance=tolerance, print_progress=True
            )
            # testing
            preds_te_count = predict_class(test_X_tf_idf_vec, weights)
            accuracy_score = accuracy_score(test_label, preds_te_count)
            precision_score = precision_score(test_label, preds_te_count)
            recall_score = recall_score(test_label, preds_te_count)
            f1_score = f1_score(test_label, preds_te_count)

            logging.log(
                level=logging.INFO,
                msg="Accuracy:{} Precision:{} Recall:{} F1-Score:{}".format(
                    accuracy_score, precision_score, recall_score, f1_score)
            )

            # print('Accuracy:', accuracy_score(test_label, preds_te_count))
            # print('Precision:', precision_score(test_label, preds_te_count))
            # print('Recall:', recall_score(test_label, preds_te_count))
            # print('F1-Score:', f1_score(test_label, preds_te_count))
