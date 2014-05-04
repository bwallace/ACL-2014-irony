'''
Code to reproduce the analyses in our ACL 2014 paper: 

    Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)
        Byron C Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak

Made possible by support from the Army Research Office (ARO), grant# 528674 
"Sociolinguistically Informed Natural Lanuage Processing: Automating Irony Detection"

Contact: Byron Wallace (byron.wallace@gmail.com)

The main methods of interest are context_stats and ml_bow. 
'''

''' built-ins. '''
import pdb
import sys
import collections
from collections import defaultdict
import re
import itertools
import sqlite3

''' dependencies: sklearn, numpy, statsmodels '''
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import statsmodels.api as sm

### assumes the database file is local!
# download this from: 
# email me (byron.wallace@gmail.com) if this url
# fails.
db_path = "ironate.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

comment_sep_str = "\n\n"+"-"*50+"\n"

def _make_sql_list_str(ls):
    return "(" + ",".join([str(x_i) for x_i in ls]) + ")"

labelers_of_interest = [2,4,5,6]
labeler_id_str = _make_sql_list_str(labelers_of_interest)

def _grab_single_element(result_set, COL=0):
    return [x[COL] for x in result_set]

def get_all_comment_ids():
    return _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where labeler_id in %s;''' % 
                    labeler_id_str)) 

def get_ironic_comment_ids():
    cursor.execute(
        '''select distinct comment_id from irony_label 
            where forced_decision=0 and label=1 and labeler_id in %s;''' % 
            labeler_id_str)

    ironic_comments = _grab_single_element(cursor.fetchall())
    return ironic_comments

def context_stats():
    '''
    Section 4, Eq (1) in the paper.

    > irony_stats.context_stats()
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3550
    Model:                          Logit   Df Residuals:                     3548
    Method:                           MLE   Df Model:                            1
    Date:                Sat, 26 Apr 2014   Pseudo R-squ.:                 0.06012
    Time:                        05:39:34   Log-Likelihood:                -2240.5
    converged:                       True   LL-Null:                       -2383.8
                                            LLR p-value:                 2.670e-64
    ==============================================================================
                     coef    std err          z      P>|z|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------
    const         -0.7108      0.040    -17.961      0.000        -0.788    -0.633
    x1             1.5081      0.093     16.223      0.000         1.326     1.690
    ==============================================================================
    '''
    all_comment_ids = get_all_comment_ids()

    # pre-context / forced decisions
    forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    for labeler in labelers_of_interest:
        labeler_forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id = %s;''' % 
                    labeler))

        all_labeler_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=0 and labeler_id = %s;''' % 
                    labeler))

        p_labeler_forced = float(len(labeler_forced_decisions))/float(len(all_labeler_decisions))
        print "labeler %s: %s" % (labeler, p_labeler_forced)

    p_forced = float(len(forced_decisions)) / float(len(all_comment_ids))

    # now look at the proportion forced for the ironic comments
    ironic_comments = get_ironic_comment_ids()
    ironic_ids_str = _make_sql_list_str(ironic_comments)
    forced_ironic_ids =  _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where 
                        forced_decision=1 and comment_id in %s and labeler_id in %s;''' % 
                                (ironic_ids_str, labeler_id_str))) 

    ''' regression bit: construct target vector + design matrix  '''
    X,y = [],[]

    for c_id in all_comment_ids:
        if c_id in forced_decisions:
            y.append(1.0)
        else:
            y.append(0.0)

        if c_id in ironic_comments:
            X.append([1.0])
        else:
            X.append([0.0])

    X = sm.add_constant(X, prepend=True)
    logit_mod = sm.Logit(y, X)
    logit_res = logit_mod.fit()
    
    print logit_res.summary()
    return logit_res

def ml_bow(show_features=False):
    '''
    Section 5, Eq (2) in the paper. 

    > irony_stats.ml_bow()
    Optimization terminated successfully.
             Current function value: 0.611578
             Iterations 5
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1949
    Model:                          Logit   Df Residuals:                     1946
    Method:                           MLE   Df Model:                            2
    Date:                Sun, 04 May 2014   Pseudo R-squ.:                 0.06502
    Time:                        08:24:43   Log-Likelihood:                -1192.0
    converged:                       True   LL-Null:                       -1274.9
                                            LLR p-value:                 9.956e-37
    ==============================================================================
                     coef    std err          z      P>|z|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------
    const         -1.3284      0.088    -15.170      0.000        -1.500    -1.157
    x1             0.9404      0.108      8.723      0.000         0.729     1.152
    x2             0.7573      0.106      7.149      0.000         0.550     0.965
    ==============================================================================

    TWO NOTES:
    1 A small bug in the original SQL code here resulted in a slightly different value for 
    x2; however the resutls are qualitatively the same as in the paper.
    2 In any case, this result will vary slightly because we are using stochastic gradient 
    descent! Still, the x2 estimate and CI (which is of interest) should be quite close.
    '''
    all_comment_ids = get_labeled_thrice_comments()

    ironic_comment_ids = get_ironic_comment_ids()
    #ironic_ids_str = _make_sql_list_str(ironic_comments)

    forced_decision_ids = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    comment_texts, y = [], []
    for id_ in all_comment_ids:
        comment_texts.append(grab_comments([id_])[0])
        if id_ in ironic_comment_ids:
            y.append(1)
        else:
            y.append(-1)

    # adding some features here; just adding them as tokens,
    # which is admittedly kind of hacky.
    emoticon_RE_str = '(?::|;|=)(?:-)?(?:\)|\(|D|P)'
    question_mark_RE_str = '\?'
    exclamation_point_RE_str = '\!'
    # any combination of multiple exclamation points and question marks
    interrobang_RE_str = '[\?\!]{2,}'

    for i, comment in enumerate(comment_texts):
        #pdb.set_trace()
        if len(re.findall(r'%s' % emoticon_RE_str, comment)) > 0:
            comment = comment + " PUNCxEMOTICON"
        if len(re.findall(r'%s' % exclamation_point_RE_str, comment)) > 0:
            comment = comment + " PUNCxEXCLAMATION_POINT"
        if len(re.findall(r'%s' % question_mark_RE_str, comment)) > 0:
            comment = comment + " PUNCxQUESTION_MARK"
        if len(re.findall(r'%s' % interrobang_RE_str, comment)) > 0:
            comment = comment + " PUNCxINTERROBANG"
        
        if any([len(s) > 2 and str.isupper(s) for s in comment.split(" ")]):
            comment = comment + " PUNCxUPPERCASE" 
        
        comment_texts[i] = comment
    # vectorize
    vectorizer = CountVectorizer(max_features=50000, ngram_range=(1,2), binary=True, stop_words="english")
    X = vectorizer.fit_transform(comment_texts)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    X_context, y_mistakes = [], []
    recalls, precisions = [], []
    Fs = []
    top_features = []
    for train, test in kf:
        train_ids = _get_entries(all_comment_ids, train)
        test_ids = _get_entries(all_comment_ids, test)
        y_train = _get_entries(y, train)
        y_test = _get_entries(y, test)

        X_train, X_test = X[train], X[test]
        svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="auto", alpha=.01)
        #pdb.set_trace()
        parameters = {'alpha':[.001, .01,  .1]}
        clf = GridSearchCV(svm, parameters, scoring='f1')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        #precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_test, preds)
        tp, fp, tn, fn = 0,0,0,0
        N = len(preds)

        for i in xrange(N):
            cur_id = test_ids[i]
            irony_indicator = 1 if cur_id in ironic_comment_ids else 0
            forced_decision_indicator = 1 if cur_id in forced_decision_ids else 0
            # so x1 is the coefficient for forced decisions (i.e., context); 
            # x2 is the coeffecient for irony (overall)
            X_context.append([irony_indicator, forced_decision_indicator])

            y_i = y_test[i]
            pred_y_i = preds[i]

            if y_i == 1:
                # ironic
                if pred_y_i == 1:
                    # true positive
                    tp += 1 
                    y_mistakes.append(0)
                else:
                    # false negative
                    fn += 1
                    y_mistakes.append(1)
            else:
                # unironic
                if pred_y_i == -1:
                    # true negative
                    tn += 1
                    y_mistakes.append(0)
                else:
                    # false positive
                    fp += 1
                    y_mistakes.append(1)

        recall = tp/float(tp + fn)
        precision = tp/float(tp + fp)
        recalls.append(recall)
        precisions.append(precision)
        f1 = 2* (precision * recall) / (precision + recall)
        Fs.append(f1)

    X_context = sm.add_constant(X_context, prepend=True)
    logit_mod = sm.Logit(y_mistakes, X_context)
    logit_res = logit_mod.fit()

    print logit_res.summary()

def grab_comments(comment_id_list, verbose=False):
    comments_list = []
    for comment_id in comment_id_list:
        cursor.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
        segments = _grab_single_element(cursor.fetchall())
        comment = " ".join(segments)
        if verbose:
            print comment
        comments_list.append(comment.encode('utf-8').strip())
    return comments_list

def _get_entries(a_list, indices):
    return [a_list[i] for i in indices]

def get_labeled_thrice_comments():
    ''' get all ids for comments labeled >= 3 times '''
    cursor.execute(
        '''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;'''
    )
    thricely_labeled_comment_ids = _grab_single_element(cursor.fetchall())
    return thricely_labeled_comment_ids




