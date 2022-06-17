import time
import numpy as np
from utils import *
from mip import CutType, OptimizationStatus
import tensorflow.compat.v1 as tf
import pandas as pd #ML package

def test_cut(model, cp):
    model.verbose = 0
    model.optimize(relax=True)
    model.verbose = 1
    # print("Original LP bound: {}".format(model.objective_value))

    improvement = []

    for cut in cp.cuts:
        m2 = model.copy()
        m2.verbose = 0
        m2 += cut
        m2.optimize(relax=True)
        improvement_ratio = (
            abs(m2.objective_value - model.objective_value)
            / max(abs(m2.objective_value), abs(model.objective_value))
        )
        improvement.append(improvement_ratio)

    return improvement


def generate_cut(model, cut_types=CutType):
    start_time = time.time()
    model.verbose = 0
    model.optimize(relax=True)
    assert (model.status == OptimizationStatus.OPTIMAL)
    #cp = model.generate_cuts([cut_types.GOMORY, cut_types.LIFT_AND_PROJECT, cut_types.RED_SPLIT, cut_types.GMI])
    #cp = model.generate_cuts([cut_types.RED_SPLIT_G])
    #cp = model.generate_cuts([cut_types.GOMORY])
    cp = model.generate_cuts(cut_types)
    sol = np.array([v.x for v in model.vars])
    end_time = time.time()
    time_cut_generated = end_time - start_time
    print(f"{len(cp.cuts)} cuts found in {time_cut_generated:.3f} s")
    model.verbose = 1
    return cp, sol


def choose_cut(cp, name='set_cover', method="Random", **kwargs):
    """
    Choose cuts according to cut features.

    :param cp: 
    """
    start_time = time.time()
    bag_features = None

    if method == "Random":
        random.seed(kwargs.get("seed"))
        ratio = kwargs.get("ratio", 0.05)
        num_cut = int(ratio * len(cp.cuts))
        selected_cuts = random.sample(cp.cuts, num_cut)
        print(f"Randomly selects {len(selected_cuts)} cuts")

    elif method == "CollectLabels":
        rng = kwargs.get("rng", np.random.RandomState())
        ratio = kwargs.get("ratio", 0.05)
        num_cut = int(ratio * len(cp.cuts))
        choose_cut_idx = rng.choice(np.arange(len(cp.cuts)), num_cut, replace=False)

        print('Index of selected cuts =', choose_cut_idx)
        choose_cut_idx = choose_cut_idx.astype(np.int32)
        selected_cuts = [cp.cuts[i] for i in choose_cut_idx]
        print('Selected cuts =', cp.cuts)

        A = kwargs.get("A", None)
        c = kwargs.get("c", None)
        x = kwargs.get("x", None)
        relax_sol = kwargs.get("relax_sol", None)
        m = kwargs.get("model", None)
        logging = kwargs.get("logging", 0)

        if logging: s = time.time()
        cut_coeff = np.array([[cut.expr.get(x_key, 0) for x_key in x] for cut in cp.cuts])
        cut_const = np.array([cut.const for cut in cp.cuts])
        cut_presence = (cut_coeff != 0).astype(np.int32)
        ''' Feature 0: support'''
        ratio_vars = np.mean(cut_presence, axis=1)
        var_fractional = (relax_sol - relax_sol.astype(np.int32) < 1e-6).astype(np.int32)
        ''' Feature 1: integral support '''
        cut_var_fractional = np.array([np.sum(var_fractional[cut_presence[i] != 0]) for i in range(cut_presence.shape[0])]) / float(A.shape[1])
        #print(ratio_vars, cut_var_fractional)
        support_features = np.concatenate((ratio_vars.reshape(-1, 1), cut_var_fractional.reshape(-1, 1)), axis=1)

        if logging: print(f"\tBasic info of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_coeff = np.mean(cut_coeff, axis=1)
        min_coeff = np.min(cut_coeff, axis=1)
        max_coeff = np.max(cut_coeff, axis=1)
        std_coeff = np.std(cut_coeff, axis=1)
        #print(mean_coeff[:5], min_coeff[:5], max_coeff[:5], std_coeff[:5])
        ''' Feature 2-5: Coeff. features'''
        coeff_features = np.concatenate((mean_coeff.reshape(-1, 1), min_coeff.reshape(-1, 1), max_coeff.reshape(-1, 1), std_coeff.reshape(-1, 1)), axis=1)
        if logging: print(f"\tCoefficients stats of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        ''' Feature 6: Parallelism'''
        cosine_with_obj = np.array([cosine_similarity(cut_coeff[i], c) for i in range(cut_coeff.shape[0])])
        #print(cosine_with_obj)
        if logging: print(f"\tCosine distance from cuts to obj coeff get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_obj = np.array([np.mean(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        min_obj = np.array([np.min(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        max_obj = np.array([np.max(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        std_obj = np.array([np.std(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        #print(mean_obj[:5], min_obj[:5], max_obj[:5], std_obj[:5])
        ''' Feature 7-10: Obj. Features '''
        obj_features = np.concatenate((mean_obj.reshape(-1, 1), min_obj.reshape(-1, 1), max_obj.reshape(-1, 1), std_obj.reshape(-1, 1)), axis=1)

        if logging: s = time.time()
        epsilon = 1e-6
        ''' Feature 11: normalized violation '''
        normalized_violation = np.array([cut.violation / (cut.const + 1e-6) for cut in cp.cuts])
        #print(normalized_violation)
        if logging: print(f"\tNormalized violation get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        ''' Feature 12: distance '''
        solution_distance = np.array([euclidean_distance(relax_sol, cut_coeff[i], -cut_const[i])
                             for i in range(cut_coeff.shape[0])])
        #print(solution_distance)
        if logging: print(f"\tSolution distance get, consuming {time.time() - s:.3f} s")

        ''' Feature 13: expected improvement '''
        expected_improvement = solution_distance * np.sqrt(np.sum(np.square(c)))

        features = np.concatenate((support_features, coeff_features, cosine_with_obj.reshape(-1, 1), obj_features,
                                   normalized_violation.reshape(-1, 1), solution_distance.reshape(-1, 1), expected_improvement.reshape(-1, 1)), axis=1)
        bag_features = features[choose_cut_idx]
        #print(bag_features[0])
        if logging: print(f"\tCoefficients stats of active obj coeff get, consuming {time.time() - s:.3f} s")

        print(bag_features)

    elif method == "DCRA_KMeans":
        cut_start_time = time.time()
        ratio = kwargs.get("ratio", 0.05)
        numOfCuts = len(cp.cuts)
        numOfSelectedCuts = int(ratio * numOfCuts) #Number of clusters: K

        if numOfCuts == 0:
            return cp.cuts, np.array([])

        A = kwargs.get("A", None)
        c = kwargs.get("c", None)
        x = kwargs.get("x", None)
        relax_sol = kwargs.get("relax_sol", None)
        m = kwargs.get("model", None)
        logging = kwargs.get("logging", 0)

        if logging: s = time.time()
        cut_coeff = np.array([[cut.expr.get(x_key, 0) for x_key in x] for cut in cp.cuts])
        cut_const = np.array([cut.const for cut in cp.cuts])
        cut_presence = (cut_coeff != 0).astype(np.int32)

        ''' Feature 0: support'''
        ratio_vars = np.mean(cut_presence, axis=1)
        var_fractional = (relax_sol - relax_sol.astype(np.int32) < 1e-6).astype(np.int32)

        ''' Feature 1: integral support '''
        cut_var_fractional = np.array(
            [np.sum(var_fractional[cut_presence[i] != 0]) for i in range(cut_presence.shape[0])]) / float(A.shape[1])
        # print(ratio_vars, cut_var_fractional)
        support_features = np.concatenate((ratio_vars.reshape(-1, 1), cut_var_fractional.reshape(-1, 1)), axis=1)

        if logging: print(f"\tBasic info of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_coeff = np.mean(cut_coeff, axis=1)
        min_coeff = np.min(cut_coeff, axis=1)
        max_coeff = np.max(cut_coeff, axis=1)
        std_coeff = np.std(cut_coeff, axis=1)
        # print(mean_coeff[:5], min_coeff[:5], max_coeff[:5], std_coeff[:5])

        ''' Feature 2-5: Coeff. features'''
        coeff_features = np.concatenate(
            (mean_coeff.reshape(-1, 1), min_coeff.reshape(-1, 1), max_coeff.reshape(-1, 1), std_coeff.reshape(-1, 1)),
            axis=1)
        if logging: print(f"\tCoefficients stats of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()

        ''' Feature 6: Parallelism'''
        cosine_with_obj = np.array([cosine_similarity(cut_coeff[i], c) for i in range(cut_coeff.shape[0])])
        # print(cosine_with_obj)
        if logging: print(f"\tCosine distance from cuts to obj coeff get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_obj = np.array([np.mean(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        min_obj = np.array([np.min(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        max_obj = np.array([np.max(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        std_obj = np.array([np.std(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        # print(mean_obj[:5], min_obj[:5], max_obj[:5], std_obj[:5])

        ''' Feature 7-10: Obj. Features '''
        obj_features = np.concatenate(
            (mean_obj.reshape(-1, 1), min_obj.reshape(-1, 1), max_obj.reshape(-1, 1), std_obj.reshape(-1, 1)), axis=1)

        if logging: s = time.time()
        epsilon = 1e-6

        ''' Feature 11: normalized violation '''
        normalized_violation = np.array([cut.violation / (cut.const + 1e-6) for cut in cp.cuts])
        # print(normalized_violation)
        if logging: print(f"\tNormalized violation get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        ''' Feature 12: distance '''
        solution_distance = np.array([euclidean_distance(relax_sol, cut_coeff[i], -cut_const[i])
                                      for i in range(cut_coeff.shape[0])])
        # print(solution_distance)
        if logging: print(f"\tSolution distance get, consuming {time.time() - s:.3f} s")

        ''' Feature 13: expected improvement '''
        expected_improvement = solution_distance * np.sqrt(np.sum(np.square(c)))

        features = np.concatenate((support_features, coeff_features, cosine_with_obj.reshape(-1, 1), obj_features,
                                   normalized_violation.reshape(-1, 1), solution_distance.reshape(-1, 1),
                                   expected_improvement.reshape(-1, 1)), axis=1)

        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)

        ''' check for zero division'''
        zero_idx = []
        for j in range(features_std.shape[0]):
            if np.abs(features_std[j]) < 1e-8:
                features_std[j] = 1
                zero_idx.append(j)
        zero_idx = np.array(zero_idx)

        features_normalized = (features - features_mean) / features_std
        if not zero_idx.shape[0] == 0:
            print("zero std detected")
            features_normalized[:, zero_idx] = 0

        '''Kmeans Clustering:'''
        from sklearn.cluster import KMeans
        features_for_kmeans = np.concatenate((coeff_features,obj_features),axis=1)
        kmeans = KMeans(int(0.05*len(features_for_kmeans)), init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(features_for_kmeans)
        print("KMeans Clustering = ", y_kmeans)
        cut_set = np.vstack((cp.cuts,y_kmeans))
        print("Cut Set = ", cut_set)

        # '''DCRA:'''
        # from mip import Model, xsum, maximize, BINARY
        # features_for_DCRA = np.concatenate((support_features, cosine_with_obj.reshape(-1, 1),
        #             normalized_violation.reshape(-1, 1), solution_distance.reshape(-1, 1),
        #             expected_improvement.reshape(-1, 1)), axis=1)

        # # Range:
        # numAlternative = len(features_for_DCRA)
        # numRank = len(features_for_DCRA)
        #
        # # Model:
        # m_rank = Model("DCRA")
        #
        # # Decision Variable:
        # x = [[m_rank.add_var(var_type=BINARY) for i in range(numAlternative)] for j in range(numRank)]
        #
        # # Objective Function:
        # m_rank.objective = maximize(
        #     xsum(features_for_DCRA[i][j] * x[i][j] for i in range(numAlternative) for j in range(numRank)))
        #
        # # Constraints:
        # for i in range(numAlternative):
        #     m_rank += xsum(x[i][j] for j in range(numRank)) == 1
        #
        # for j in range(numRank):
        #     m_rank += xsum(x[i][j] for i in range(numAlternative)) == 1
        #
        # # Optimizing:
        # m_rank.optimize()
        #
        # # Print Ouput:
        # print("DCRA Rank:")
        # for i in range(numAlternative):
        #     for j in range(numRank):
        #         if x[i][j].x >= 0.99:
        #             print("Alternative {} is rank at {}".format(i, j))

    elif method == "Test":
        cut_start_time = time.time()
        random.seed(kwargs.get("seed"))
        ratio = kwargs.get("ratio", 0.05)
        numOfCuts = len(cp.cuts)
        numOfSubCuts = int(ratio * numOfCuts)
        if numOfCuts == 0:
            return cp.cuts, np.array([])
        #print("number of sub cuts is ", numOfSubCuts)
        A = kwargs.get("A", None)
        c = kwargs.get("c", None)
        x = kwargs.get("x", None)
        relax_sol = kwargs.get("relax_sol", None)
        m = kwargs.get("model", None)
        logging = kwargs.get("logging", 0)

        if logging: s = time.time()
        cut_coeff = np.array([[cut.expr.get(x_key, 0) for x_key in x] for cut in cp.cuts])
        cut_const = np.array([cut.const for cut in cp.cuts])
        cut_presence = (cut_coeff != 0).astype(np.int32)
        ''' Feature 0: support'''
        ratio_vars = np.mean(cut_presence, axis=1)
        var_fractional = (relax_sol - relax_sol.astype(np.int32) < 1e-6).astype(np.int32)
        ''' Feature 1: integral support '''
        cut_var_fractional = np.array(
            [np.sum(var_fractional[cut_presence[i] != 0]) for i in range(cut_presence.shape[0])]) / float(A.shape[1])
        # print(ratio_vars, cut_var_fractional)
        support_features = np.concatenate((ratio_vars.reshape(-1, 1), cut_var_fractional.reshape(-1, 1)), axis=1)

        if logging: print(f"\tBasic info of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_coeff = np.mean(cut_coeff, axis=1)
        min_coeff = np.min(cut_coeff, axis=1)
        max_coeff = np.max(cut_coeff, axis=1)
        std_coeff = np.std(cut_coeff, axis=1)
        # print(mean_coeff[:5], min_coeff[:5], max_coeff[:5], std_coeff[:5])
        ''' Feature 2-5: Coeff. features'''
        coeff_features = np.concatenate(
            (mean_coeff.reshape(-1, 1), min_coeff.reshape(-1, 1), max_coeff.reshape(-1, 1), std_coeff.reshape(-1, 1)),
            axis=1)
        if logging: print(f"\tCoefficients stats of cuts get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        ''' Feature 6: Parallelism'''
        cosine_with_obj = np.array([cosine_similarity(cut_coeff[i], c) for i in range(cut_coeff.shape[0])])
        # print(cosine_with_obj)
        if logging: print(f"\tCosine distance from cuts to obj coeff get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        mean_obj = np.array([np.mean(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        min_obj = np.array([np.min(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        max_obj = np.array([np.max(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        std_obj = np.array([np.std(c[cut_coeff[i] != 0]) for i in range(cut_coeff.shape[0])])
        # print(mean_obj[:5], min_obj[:5], max_obj[:5], std_obj[:5])
        ''' Feature 7-10: Obj. Features '''
        obj_features = np.concatenate(
            (mean_obj.reshape(-1, 1), min_obj.reshape(-1, 1), max_obj.reshape(-1, 1), std_obj.reshape(-1, 1)), axis=1)

        if logging: s = time.time()
        epsilon = 1e-6
        ''' Feature 11: normalized violation '''
        normalized_violation = np.array([cut.violation / (cut.const + 1e-6) for cut in cp.cuts])
        # print(normalized_violation)
        if logging: print(f"\tNormalized violation get, consuming {time.time() - s:.3f} s")

        if logging: s = time.time()
        ''' Feature 12: distance '''
        solution_distance = np.array([euclidean_distance(relax_sol, cut_coeff[i], -cut_const[i])
                                      for i in range(cut_coeff.shape[0])])
        # print(solution_distance)
        if logging: print(f"\tSolution distance get, consuming {time.time() - s:.3f} s")

        ''' Feature 13: expected improvement '''
        expected_improvement = solution_distance * np.sqrt(np.sum(np.square(c)))

        features = np.concatenate((support_features, coeff_features, cosine_with_obj.reshape(-1, 1), obj_features,
                                   normalized_violation.reshape(-1, 1), solution_distance.reshape(-1, 1),
                                   expected_improvement.reshape(-1, 1)), axis=1)
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        ''' check for zero division'''
        zero_idx = []
        for j in range(features_std.shape[0]):
            if np.abs(features_std[j]) < 1e-8:
                features_std[j] = 1
                zero_idx.append(j)
        zero_idx = np.array(zero_idx)

        features_normalized = (features - features_mean) / features_std
        if not zero_idx.shape[0] == 0:
            print("zero std detected")
            features_normalized[:, zero_idx] = 0

        if logging: print(f"\tCoefficients stats of active obj coeff get, consuming {time.time() - s:.3f} s")
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(f'./saved_model/{name}/model.meta')       # model path
            new_saver.restore(sess, tf.train.latest_checkpoint(f'./saved_model/{name}/'))
            graph = tf.get_default_graph()
            input_x = graph.get_tensor_by_name("input_x:0")
            output = graph.get_tensor_by_name("output:0")
            logits = sess.run([output], {input_x: features_normalized})
            logits = np.array(logits).reshape(numOfCuts, -1)
            # print(scores.shape)
            e_x = np.exp(logits - np.max(logits, axis=1).reshape(-1, 1))
            probs = e_x / np.sum(e_x, axis=1).reshape(-1, 1)
            scores = probs[:, 0]

            scores_cut_set = zip(scores, cp.cuts)
            scores_sorted_cut_set = sorted(scores_cut_set, key=lambda x: abs(x[0]), reverse=True)
            selected_cuts = [cut[1] for cut in scores_sorted_cut_set[:numOfSubCuts]]
            bag_features = features[:numOfSubCuts]
            print("time for selecting cuts = ", time.time() - cut_start_time)
    else:
        raise NotImplementedError

    end_time = time.time()
    if logging:
        print(f"{len(selected_cuts)} cuts found in {end_time - start_time:.3f}")
    return selected_cuts, bag_features
