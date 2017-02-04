# -*- coding: utf-8 -*-

import LogisticRegession as lr
import numpy as np

def classify_vector(in_x, weights):
    prob = lr.sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colic_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    # 组织训练数据
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        # 训练样本
        training_set.append(line_arr)
        # 训练标签
        training_labels.append(float(curr_line[21]))
    # 训练后得到的系数
    training_weights = lr.stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    error_count = 0; num_test_vec = 0.0
    # 测试该算法的错误率
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []

        for i in range(21):
            line_arr.append(float(curr_line[i]))

        if int(classify_vector(np.array(line_arr), training_weights)) != \
                int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / num_test_vec)
    print "the error rate of this test is : %f " % error_count
    return error_rate

def multi_test():
    num_tests = 10; error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print "after %d iterations the average error rate " \
          "is : %f" % (num_tests, error_sum / float(num_tests))

if __name__ == '__main__':
    multi_test()