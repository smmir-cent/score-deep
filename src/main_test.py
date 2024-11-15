import pandas as pd
import numpy as np
from tabgan.src.tabgan.utils import get_year_mnth_dt_from_date,make_two_digit,collect_dates
from tabgan.src.tabgan.sampler import OriginalGenerator, GANGenerator

if __name__ == "__main__":

    train_size = 100
    train = pd.DataFrame(
            np.random.randint(-10, 150, size=(train_size, 4)), columns=list("ABCD")
        )
    min_date = pd.to_datetime('2019-01-01')
    max_date = pd.to_datetime('2021-12-31')
    d = (max_date - min_date).days + 1

    train['Date'] = min_date + pd.to_timedelta(pd.np.random.randint(d, size=train_size), unit='d')
    train = get_year_mnth_dt_from_date(train, 'Date')
    print("before GANGenerator")
    new_train, new_target = GANGenerator(gen_x_times=1.1, cat_cols=['year'], bot_filter_quantile=0.001,
                                        top_filter_quantile=0.999,
                                        is_post_process=True, pregeneration_frac=2, only_generated_data=False).\
                                        generate_data_pipe(train.drop('Date', axis=1), None,
                                                            train.drop('Date', axis=1))
                                                    
    print("before collect_dates")
    new_train = collect_dates(new_train)
