import numpy as np

class user():
    def __init__(self, alpha=0.5):
        '''
        Assigns the sensitivity of a user.
        We consider a user that consumes one or more contents during a session, 
        drawn from a content catalogue. 
        With probability alpha she follows one of the N recommendations.
        With probability 1 − alpha (alpha ∈ [0, 1]) she ignores the recommendations. 

        Input: alpha: float
        '''
        self.alpha = alpha
        # TODO: maybe add a vector with the preferences of the user for the different contents (this maybe should take as input the content catalogue); You can leave it as is now, but fix later

    def characteristics(self):
        ''' 
        Output: returns specs of user
        '''
        return 'User\'s preference on RSs: {}'.format(self.alpha)

    def getAlpha(self):
        return self.alpha

    def decision(self):
        # TODO: this function should receive as input also the (a) recommendation list and (b) the content catalogue, and return the content which the user selects next.
        return np.random.choice(
            ['Recommendation List', 'Search Bar'],
            p=[self.getAlpha(), 1 - self.getAlpha()]
        )


def main():
    u = user()
    print(u.characteristics())


if __name__ == "__main__":
    main()
