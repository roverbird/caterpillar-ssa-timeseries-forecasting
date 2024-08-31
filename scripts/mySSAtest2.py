import numpy as np
import pandas as pd
from numpy import matrix as m
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 11, 4

class mySSA(object):
    '''Singular Spectrum Analysis object'''
    def __init__(self, time_series):
        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq

    @staticmethod
    def _printer(name, *args):
        '''Helper function to print messages neatly'''
        print('-'*40)
        print(name+':')
        for msg in args:
            print(msg)  
    
    @staticmethod
    def get_contributions(X=None, s=None, plot=True):
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(s, 2)
        frob_norm = np.linalg.norm(X)
        ret = pd.DataFrame(lambdas / (frob_norm ** 2), columns=['Contribution'])
        ret['Contribution'] = ret.Contribution.round(4)
        if plot:
            ax = ret[ret.Contribution != 0].plot.bar(legend=False)
            ax.set_xlabel("Lambda_i")
            ax.set_title('Non-zero contributions of Lambda_i')
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
            plt.savefig('contributions_plot.png')  # Save plot
            plt.close()  # Close plot to avoid displaying
            return ax
        return ret[ret.Contribution > 0]
    
    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []
        
        # Diagonal Averaging
        for k in range(1 - K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1 - mask)
            ret += [ma.sum() / mask_n]
        
        return pd.DataFrame(ret).rename(columns={0: 'Reconstruction'})
    
    def split_data(self, train_size=0.8):
        '''Split the time series into training and testing sets'''
        split_index = int(self.ts_N * train_size)
        self.train_ts = self.ts[:split_index]
        self.test_ts = self.ts[split_index:]
        self.train_ts_v = self.train_ts.values
        self.test_ts_v = self.test_ts.values
        self.train_ts_N = self.train_ts.shape[0]
        self.test_ts_N = self.test_ts.shape[0]

    def view_time_series(self):
        '''Plot the time series'''
        ax = self.ts.plot(title='Original Time Series')
        plt.savefig('time_series_plot.png')  # Save plot
        plt.close()  # Close plot to avoid displaying
    
    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
        '''Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        ts = self.train_ts if hasattr(self, 'train_ts') else self.ts
        
        if not embedding_dimension:
            self.embedding_dimension = ts.shape[0] // 2
        else:
            self.embedding_dimension = embedding_dimension
        if suspected_frequency:
            self.suspected_frequency = suspected_frequency
            self.embedding_dimension = (self.embedding_dimension // self.suspected_frequency) * self.suspected_frequency
    
        self.K = ts.shape[0] - self.embedding_dimension + 1
        self.X = m(linalg.hankel(ts, np.zeros(self.embedding_dimension))).T[:, :self.K]
        self.X_df = pd.DataFrame(self.X)
        self.X_complete = self.X_df.dropna(axis=1)
        self.X_com = m(self.X_complete.values)
        self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)
        self.X_miss = m(self.X_missing.values)
        self.trajectory_dimentions = self.X_df.shape
        self.complete_dimensions = self.X_complete.shape
        self.missing_dimensions = self.X_missing.shape
        self.no_missing = self.missing_dimensions[1] == 0
            
        if verbose:
            msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
            msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
            msg1 = msg1.format(self.embedding_dimension, self.trajectory_dimentions)
            msg2 = msg2.format(self.complete_dimensions, self.missing_dimensions)
            self._printer('EMBEDDING SUMMARY', msg1, msg2)
        
        if return_df:
            return self.X_df
        
    def decompose(self, verbose=False):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace'''
        X = self.X_com
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys, Zs = {}, {}, {}, {}
        for i in range(self.d):
            Zs[i] = self.s[i] * self.V[:, i]
            Vs[i] = X.T * (self.U[:, i] / self.s[i])
            Ys[i] = self.s[i] * self.U[:, i]
            Xs[i] = Ys[i] * (m(Vs[i]).T)
        self.Vs, self.Xs = Vs, Xs
        self.s_contributions = self.get_contributions(X, self.s, False)
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4)
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}
        
        if verbose:
            msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
            msg1 = msg1.format(self.d, self.r)
            msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic)
            self._printer('DECOMPOSITION SUMMARY', msg1, msg2)
    
    def view_s_contributions(self, adjust_scale=False, cumulative=False, return_df=False):
        '''View the contribution to variance of each singular value and its corresponding signal'''
        contribs = self.s_contributions.copy()
        contribs = contribs[contribs.Contribution != 0]
        if cumulative:
            contribs['Contribution'] = contribs.Contribution.cumsum()
        if adjust_scale:
            contribs = (1 / contribs).max() * 1.1 - (1 / contribs)
        ax = contribs.plot.bar(legend=False)
        ax.set_xlabel("Singular_i")
        ax.set_title('Non-zero{} contribution of Singular_i {}'.\
                     format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
        if adjust_scale:
            ax.axes.get_yaxis().set_visible(False)
        else:
            vals = ax.get_yticks()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:3.0f}%'.format(x * 100)))
        plt.savefig('singular_contributions_plot.png')  # Save plot
        plt.close()  # Close plot to avoid displaying
        if return_df:
            return contribs


    @classmethod
    def view_reconstruction(cls, *hankel, names=None, return_df=False, plot=True, symmetric_plots=False):
        '''Visualize the reconstruction of the hankel matrix/matrices passed to *hankel'''
        hankel_mat = None
        for han in hankel:
            if isinstance(hankel_mat, m):
                hankel_mat = hankel_mat + han
            else: 
                hankel_mat = han.copy()
        hankel_full = cls.diagonal_averaging(hankel_mat)
        title = 'Reconstruction of signal'
        if names or names == 0: 
            title += ' associated with singular value{}: {}'
            title = title.format('' if len(str(names)) == 1 else 's', names)
        if plot:
            ax = hankel_full.plot(legend=False, title=title)
            if symmetric_plots:
                velocity = hankel_full.abs().max().iloc[0]  # Use .iloc for positional indexing
                ax.set_ylim(bottom=-velocity, top=velocity)
            plt.savefig(f'reconstruction_plot_{names}.png')  # Save plot
            plt.close()  # Close plot to avoid displaying
        if return_df:
            return hankel_full

        
    def _forecast_prep(self, singular_values=None):
        self.X_com_hat = np.zeros(self.complete_dimensions)
        self.verticality_coefficient = 0
        self.forecast_orthonormal_base = {}
        if singular_values:
            try:
                for i in singular_values:
                    self.forecast_orthonormal_base[i] = self.orthonormal_base[i]
            except:
                if singular_values == 0:
                    self.forecast_orthonormal_base[0] = self.orthonormal_base[0]
                else:
                    raise ValueError('Please pass in a list/array of singular value indices to use for forecast')
        else:
            self.forecast_orthonormal_base = self.orthonormal_base
        self.R = np.zeros(self.forecast_orthonormal_base[0].shape)[:-1]
        for Pi in self.forecast_orthonormal_base.values():
            self.X_com_hat += Pi * Pi.T * self.X_com
            pi = np.ravel(Pi)[-1]
            self.verticality_coefficient += pi ** 2
            self.R += pi * Pi[:-1]
        self.R = m(self.R / (1 - self.verticality_coefficient))
        self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)
            
    def forecast_recurrent(self, steps_ahead=12, singular_values=None, plot=False, return_df=False, allow_negative=False, **plotargs):
        '''Forecast from last point of original time series up to steps_ahead using recurrent methodology.
        Optionally applies constraints to avoid unrealistic negative or excessively large values.'''
        
        try:
            self.X_com_hat
        except AttributeError:
            self._forecast_prep(singular_values)
        
        ts = self.train_ts_v if hasattr(self, 'train_ts_v') else self.ts_v
        
        self.ts_forecast = np.array(ts[0])
        for i in range(1, len(ts) + steps_ahead):
            try:
                if np.isnan(ts[i]):
                    x = self.R.T @ m(self.ts_forecast[max(0, i - self.R.shape[0]): i]).T
                    if not allow_negative:
                        x = np.maximum(0, x[0])  # Apply non-negative constraint if allow_negative is False
                    else:
                        x = x[0]  # Allow negative values
                    self.ts_forecast = np.append(self.ts_forecast, x)
                else:
                    self.ts_forecast = np.append(self.ts_forecast, ts[i])
            except IndexError:
                x = self.R.T @ m(self.ts_forecast[i - self.R.shape[0]: i]).T
                if not allow_negative:
                    x = np.maximum(0, x[0])  # Apply non-negative constraint if allow_negative is False
                else:
                    x = x[0]  # Allow negative values
                self.ts_forecast = np.append(self.ts_forecast, x)
        
        self.forecast_N = i + 1
        new_index = pd.date_range(start=self.ts.index.min(), periods=self.forecast_N, freq=self.freq)
        forecast_df = pd.DataFrame(self.ts_forecast, columns=['Forecast'], index=new_index)
        forecast_df['Original'] = np.append(ts, [np.nan] * steps_ahead)
        

        # Plotting
        if plot:
            ax = forecast_df.plot(title='Forecasted vs. Original Time Series', **plotargs)
            
            # Improve the plot
            ax.grid(True)  # Add a grid to the plot
            ax.set_xlabel('Time')  # Set x-axis label
            ax.set_ylabel('Values')  # Set y-axis label
            
            # Adjust x-axis ticks for better readability
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # Set a maximum of 10 x-axis ticks
            
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            
            # Save the plot
            plt.savefig('forecast_plot.png')
            plt.close()

        
        if return_df:
            return forecast_df
    
    def evaluate(self):
        '''Evaluate the forecasting performance on the test set and calculate MAPE'''
        if not hasattr(self, 'test_ts'):
            raise ValueError('Data is not split into train and test sets. Call split_data() first.')
        
        # Perform SSA embedding and decomposition
        self.embed(embedding_dimension=36, suspected_frequency=6)  # Use default or adjust parameters as needed
        self.decompose()
        
        # Generate forecasts
        forecast_df = self.forecast_recurrent(steps_ahead=self.test_ts_N, plot=False, return_df=True)
        
        # Align test data with forecast
        test_df = pd.DataFrame(self.test_ts_v, columns=['Actual'], index=self.test_ts.index)
        comparison_df = test_df.join(forecast_df[['Forecast']])
        
        # Calculate MAPE
        comparison_df['Absolute Error'] = abs(comparison_df['Actual'] - comparison_df['Forecast'])
        comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Actual']) * 100
        mape = comparison_df['Percentage Error'].mean()
        
        # Save comparison plot
        #ax = comparison_df.plot(title='Forecast vs Actual')
        comparison_df[['Actual', 'Forecast', 'Absolute Error']].plot(title='Forecast vs Actual')
        plt.savefig('forecast_vs_actual.png')
        plt.close()
        
        print(f'MAPE: {mape:.2f}%')
        
        return comparison_df


if __name__=='__main__':
    # Read the CSV file without parsing dates
    ts = pd.read_csv('./data/TideHourly.csv', index_col='timestamp')

    # Convert the index to datetime using the specified format
    ts.index = pd.to_datetime(ts.index, format='%d-%m-%Y-%H%M')

    # Example of other timestamp formats, adjust as in your dataset
    #ts.index = pd.to_datetime(ts.index, format='%Y-%m-%d')
    #ts.index = pd.to_datetime(ts.index, format='%Y-%m-%d-%H-%M')
    #ts.index = pd.to_datetime(ts.index, format='%Y-%m')

    ssa = mySSA(ts)

    # Split data into training and test sets
    ssa.split_data(train_size=0.9)

    # Plot original series for reference
    ssa.view_time_series()

    # PARAMETER SETTINGS
    ssa.embed(embedding_dimension=86, suspected_frequency=7, verbose=True)

    ssa.decompose(True)
    ssa.view_s_contributions(adjust_scale=True)

    # Component Signals
    components = [i for i in range(13)]
    rcParams['figure.figsize'] = 11, 2
    for i in range(5):
        ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i != 0)
    rcParams['figure.figsize'] = 11, 4

    # Reconstruction
    ssa.view_reconstruction(*[ssa.Xs[i] for i in components], names=components)

    # Forecasting
    # Set steps ahead here (the length of forecast)
    forecast_df = ssa.forecast_recurrent(steps_ahead=14, plot=True)
    
    # Evaluate the forecasting performance
    comparison_df = ssa.evaluate()
    print(comparison_df)

