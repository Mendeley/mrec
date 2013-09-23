class Evaluator(object):
    """
    Compute metrics for recommendations that have been written to file.

    Parameters
    ----------
    compute_metrics : function(list,list)
        The evaluation function which should accept two lists of predicted
        and actual item indices.
    max_items : int
        The number of recommendations needed to compute the evaluation function.
    """

    def __init__(self,compute_metrics,max_items):
        self.compute_metrics = compute_metrics
        self.max_items = max_items

    def _add_metrics(self,predicted,actual):
        metrics = self.compute_metrics(predicted,actual)
        if metrics:
            for m,val in metrics.iteritems():
                self.cum_metrics[m] += val
            self.count += 1

    def process(self,testdata,recsfile,start,end,offset=1):
        """
        Parameters
        ----------
        testdata : scipy sparse matrix
            The test items for each user.
        recsfile : str
            Filepath to the recommendations.  The file should contain TSV
            of the form: user, item, score.  IMPORTANT: the recommendations must
            be sorted by user and score.
        start : int
            First user to evaluate.
        end: int
            One after the last user to evaluate.
        offset : int
            Index offset for users and items in recommendations file.

        Returns
        -------
        cum_metrics : dict
            Aggregated metrics i.e. total values for all users.
        count : int
            The number of users for whom metrics were computed.
        """
        from collections import defaultdict

        self.cum_metrics = defaultdict(float)
        self.count = 0

        last_user = start
        recs = []
        for line in open(recsfile):
            user,item,score = line.strip().split('\t')
            user = int(user)-1  # convert to 0-indxed
            item = int(item)-1
            if user >= end:
                break
            if user < start:
                continue
            if user != last_user:
                self._add_metrics(recs,testdata[last_user,:].indices.tolist())
                last_user = user
                recs = []
            if len(recs) < self.max_items:
                recs.append(item)
        self._add_metrics(recs,testdata[last_user,:].indices.tolist())

        return self.cum_metrics,self.count
