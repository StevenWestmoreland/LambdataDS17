''' Tests for lambdata modules. '''

import unittest
from random import randint

from example_module import increment, COLORS
from oop_example import SocialMediaUser
from dataframe_helper import My_Data_Splitter


class ExampleUnitTests(unittest.TestCase):
    ''' Making sure the examples behave as expected. '''
    def test_increment(self):
        ''' Testing increment adds 1 to a number'''
        x1 = 5
        y1 = increment(x1)
        x2 = -106
        y2 = increment(x2)
        # Now we make sure the output is as expected with assertions
        self.assertEqual(y1, 6)
        self.assertEqual(y2, -105)

    def test_increment_random(self):
        ''' Test increment with randomly generated input '''
        x1 = randint(1, 1000000)
        y1 = increment(x1)
        self.asserEqual(y1, x1 + 1)
    
    def test_colors(self):
        ''' Testing presence/absence of expected colors '''
        self.assertIn('Orange', COLORS)
        self.asserNotIn('Aquamarine', COLORS)
        self.assertEqual(len(COLORS, 6))


class SocialMediaUserTests(unittest.TestCase):
    ''' Test the instantiation and use of SocialMediaUser '''
    def test_name(self):
        ''' Test that the name field is assigned correctly '''
        user1 = SocialMediaUser('Jane')
        user2 = SocialMediaUser('Joe')
        self.assertEqual(user1.name, 'Jane')
        self.assertEqual(user2.name, 'Joe')

    def test_default_upvotes(self):
        '''Test that the default upvotes of a new user is 0 '''
        user1 = SocialMediaUser('Jane')
        self.assertEqual(user1.upvotes, 0)

    def test_unpopular(self):
        ''' Test that a user with <=100 upvotes is not popular'''
        user1 = SocialMediaUser('Joe')
        for _ in range(randint(1, 100)):
            user1.recieve_upvote()
        self.assertEqual(user1.is_popular(), False)

    def test_popular(self):
        ''' Test that a user with >100 upvotes is popular '''
        user1 = SocialMediaUser('Jand')
        for _ in range(randint(101, 10000)):
            user1.recieve_upvote()
        self.assertEqual(user1.is_popular(), True)


class DataframeHelper(unittest, TestCase):
    ''' Test the various dataframe helper methods'''
    def test_date_divider(self):
        raw_data = {'name': ['Willard Morris', 'Al Jennings', 'Omar Mullins', 'Spencer McDaniel'],
                    'age': [20, 19, 22, 21],
                    'favorite_color': ['blue', 'red', 'yellow', "green"],
                    'grade': [88, 92, 95, 70],
                    'birth_date': ['01-1996', '08-05-1997', '04-28-1996', '12-16-1995']}
        df = pd.DataFrame(raw_data, index = ['Willard Morris', 'Al Jennings', 'Omar Mullins', 'Spencer McDaniel'])
        date_col = 'birth_date'

        current_shape = df.shape[1]
        expected_shape = current_shape + 3

        # Create My_Data_Splitter object
        splitter = My_Data_Splitter(df, features=['age', 'favorite_color', 'birth_date'], target='grade')
        converted_df = splitter.date_divider(date_col)

        self.assertEqual(expected_shape, converted_df.shape[1])
        self.assertIn(20, df['age'].tolist())


if __name__ == '__main__':
    # This conditional is for code that will be run
    # when we execute our file from the command line
    unittest.main()