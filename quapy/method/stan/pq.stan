data {
  int<lower=0> n_bucket;
  array[n_bucket] int<lower=0> train_pos;
  array[n_bucket] int<lower=0> train_neg;
  array[n_bucket] int<lower=0> test;
  int<lower=0,upper=1> posterior;
}

transformed data{
  row_vector<lower=0>[n_bucket] train_pos_rv;
  row_vector<lower=0>[n_bucket] train_neg_rv;
  row_vector<lower=0>[n_bucket] test_rv;
  real n_test;

  train_pos_rv = to_row_vector( train_pos );
  train_neg_rv = to_row_vector( train_neg );
  test_rv      = to_row_vector( test );
  n_test       = sum( test );
}

parameters {
  simplex[n_bucket] p_neg;
  simplex[n_bucket] p_pos;
  real<lower=0,upper=1> prev_prior;
}

model {
  if( posterior ) {
    target += train_neg_rv * log( p_neg );
    target += train_pos_rv * log( p_pos );
    target += test_rv * log( p_neg * ( 1 - prev_prior) + p_pos * prev_prior );
  }
}

generated quantities {
  real<lower=0,upper=1> prev;
  prev = sum( binomial_rng(test, 1 / ( 1 + (p_neg./p_pos) *(1-prev_prior)/prev_prior ) ) ) / n_test;
}