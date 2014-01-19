def retrain_recommender(model,dataset):
    model.fit(dataset.X)

if __name__ == '__main__':

    try:
        from sklearn.grid_search import ParameterGrid
    except ImportError:
        from sklearn.grid_search import IterGrid as ParameterGrid
    from optparse import OptionParser
    from warp import WARPMFRecommender

    from mrec.evaluation.metrics import *

    parser = OptionParser()
    parser.add_option('-m','--main_split_dir',dest='main_split_dir',help='directory containing 50/50 splits for main evaluation')
    parser.add_option('-l','--loo_split_dir',dest='loo_split_dir',help='directory containing LOO splits for hit rate evaluation')
    parser.add_option('-n','--num_splits',dest='num_splits',type='int',default=5,help='number of splits in each directory (default: %default)')

    (opts,args) = parser.parse_args()
    if not (opts.main_split_dir or opts.loo_split_dir) or not opts.num_splits:
        parser.print_help()
        raise SystemExit

    print 'doing a grid search for regularization parameters...'
    params = {'d':[100],'gamma':[0.01],'C':[100],'max_iter':[100000],'validation_iters':[500]}
    models = [WARPMFRecommender(**a) for a in ParameterGrid(params)]

    for train in glob:
        # get test
        # load em both up
        # put them into something that returns train,test.keys(),test in a generator()
        # test is a dict id->[id,id,...]

    if opts.main_split_dir:
        generate_main_metrics = generate_metrics(get_known_items_from_dict,compute_main_metrics)
        main_metrics = run_evaluation(models,
                                      retrain_recommender,
                                      load_splits(opts.main_split_dir,opts.num_splits),
                                      opts.num_splits,
                                      generate_main_metrics)
        print_report(models,main_metrics)

    if opts.loo_split_dir:
        generate_hit_rate = generate_metrics(get_known_items_from_dict,compute_hit_rate)
        hit_rate_metrics = run_evaluation(models,
                                          retrain_recommender,
                                          load_splits(opts.loo_split_dir,opts.num_splits),
                                          opts.num_splits,
                                          generate_hit_rate)
        print_report(models,hit_rate_metrics)
