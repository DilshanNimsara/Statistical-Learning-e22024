import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as graph_objects
from plotly.subplots import make_subplots
import scipy.stats as stats
from google.colab import files

class PlottingMethods:
    """
    A modular utility class providing granular components for data visualization.
    Returns standalone, HTML-wrapped Plotly structures suitable for flexible embedding.
    """

    @staticmethod
    def generate_bar_chart(df, column, title="Bar Chart"):
        """Generates a bar chart displaying both raw counts and percentage labels."""
        if df is None or column not in df.columns or df[column].empty:
            return "<div>No data available for rendering.</div>"

        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        total = counts['Count'].sum()
        counts['Percentage'] = (counts['Count'] / total * 100).round(2)
        counts['Label'] = counts.apply(lambda r: f"{r['Count']} ({r['Percentage']}%)", axis=1)

        fig = px.bar(counts, x=column, y='Count', text='Label', title=title,
                     labels={column: str(column), 'Count': 'Frequency'})
        fig.update_traces(textposition='outside')
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    @staticmethod
    def generate_pie_chart(df, column, title="Pie Chart"):
        """Generates a structured interactive pie chart."""
        if df is None or column not in df.columns or df[column].empty:
            return "<div>No data available for rendering.</div>"

        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        fig = px.pie(counts, names=column, values='Count', title=title)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    @staticmethod
    def generate_histogram(df, column, bins=30, title="Histogram"):
        """Generates a standard distribution histogram for numerical variables."""
        if df is None or column not in df.columns or df[column].empty:
            return "<div>No data available for rendering.</div>"

        fig = px.histogram(df, x=column, nbins=bins, title=title, marginal="rug")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

class DataInspector:
    """
    An automated processing class designed for data-wrangling,
    advanced feature scaling, and high-level structural exploration.
    """
    def __init__(self):
        self.df = None
        self.garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']

    def upload_data(self):
        """Handles local file uploads directly within Google Colab runtimes."""
        uploaded = files.upload()
        if not uploaded:
            print("No file selected.")
            return None

        filename = list(uploaded.keys())[0]
        # Ingest with explicit initial parsing of common garbage representations
        self.df = pd.read_csv(io.BytesIO(uploaded[filename]), na_values=self.garbage_strings)
        self._auto_type_correction()
        print(f"Successfully loaded {filename}. Current baseline shape: {self.df.shape}")
        return self.df

    def _auto_type_correction(self):
        """Forces string-encoded numeric variables into their correct typing paradigms."""
        if self.df is None:
            return
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Attempt structural numeric parsing
                converted = pd.to_numeric(self.df[col], errors='coerce')
                # Retain structural conversion if it does not ruin the entire feature matrix
                if not converted.isna().all():
                    self.df[col] = converted

    def display_summary(self):
        """Displays data matrix shape, type segment counts, and previews data."""
        if self.df is None:
            print("Engine holds no active DataFrame context.")
            return

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print("="*60)
        print(f"Data Shape: {self.df.shape[0]} rows | {self.df.shape[1]} columns")
        print(f"Numerical Attributes ({len(num_cols)}): {num_cols}")
        print(f"Categorical Attributes ({len(cat_cols)}): {cat_cols}")
        print("="*60)
        print("\nFirst 20 Matrix Rows Preview:")
        display(self.df.head(20))

    def handle_missing_values(self, strategies=None):
        """
        Implements strategic data imputation rules.
        strategies: dict pairing column names with rules ('mean', 'median', 'mode', or constant values)
        """
        if self.df is None or not strategies:
            return self.df

        for col, strategy in strategies.items():
            if col not in self.df.columns:
                continue

            if self.df[col].isna().sum() == 0:
                continue

            if strategy == 'mean':
                val = self.df[col].mean()
            elif strategy == 'median':
                val = self.df[col].median()
            elif strategy == 'mode':
                val = self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan
            else:
                val = strategy # Custom constant fallback

            self.df[col] = self.df[col].fillna(val)
        return self.df

    def remove_duplicates(self):
        """Removes duplicate rows from the active data structure."""
        if self.df is not None:
            init_rows = self.df.shape[0]
            self.df.drop_duplicates(inplace=True)
            print(f"Pruned {init_rows - self.df.shape[0]} duplicate rows.")
        return self.df

    def handle_outliers(self, columns, action='flag'):
        """
        Identifies outliers using an IQR-based system.
        action options: 'flag' (adds a boolean outlier column) or 'delete' (drops outlier rows)
        """
        if self.df is None or not columns:
            return self.df

        mask = pd.Series(False, index=self.df.index)
        for col in columns:
            if col in self.df.columns and np.issubdtype(self.df[col].dtype, np.number):
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = mask | (self.df[col] < lower) | (self.df[col] > upper)

        if action == 'delete':
            init_shape = self.df.shape[0]
            self.df = self.df[~mask].reset_index(drop=True)
            print(f"Dropped {init_shape - self.df.shape[0]} outlier rows.")
        elif action == 'flag':
            self.df['is_outlier'] = mask
            print(f"Flagged {mask.sum()} systemic outlier rows in 'is_outlier'.")
        return self.df

    def delete_rows(self):
        """Interactive pruning method that accepts comma-separated row indices to drop."""
        if self.df is None: return
        user_input = input("Enter comma-separated integer index values to delete: ")
        try:
            indices = [int(x.strip()) for x in user_input.split(',') if x.strip()]
            self.df.drop(index=indices, inplace=True, errors='ignore')
            self.df.reset_index(drop=True, inplace=True)
            print(f"Successfully pruned specified rows. Current shape: {self.df.shape}")
        except ValueError:
            print("Invalid index array parsing error.")

    def delete_columns(self):
        """Interactive pruning method that accepts comma-separated column names to drop."""
        if self.df is None: return
        user_input = input("Enter comma-separated column names to delete: ")
        cols = [x.strip() for x in user_input.split(',') if x.strip()]
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        print(f"Successfully pruned specified columns. Current shape: {self.df.shape}")

    def extract_normalized_numeric_data(self, columns, method='standard'):
        """Scales numerical features using standard, minmax, or robust scaling methods."""
        if self.df is None or not columns:
            return pd.DataFrame()

        sub_df = self.df[columns].copy()
        for col in columns:
            if method == 'minmax':
                c_min, c_max = sub_df[col].min(), sub_df[col].max()
                sub_df[col] = (sub_df[col] - c_min) / (c_max - c_min) if c_max != c_min else 0
            elif method == 'standard':
                mean, std = sub_df[col].mean(), sub_df[col].std()
                sub_df[col] = (sub_df[col] - mean) / std if std != 0 else 0
            elif method == 'robust':
                q1, median, q3 = sub_df[col].quantile(0.25), sub_df[col].median(), sub_df[col].quantile(0.75)
                iqr = q3 - q1
                sub_df[col] = (sub_df[col] - median) / iqr if iqr != 0 else 0
        return sub_df.add_prefix(f'{method}_')

    def extract_normalized_categorical_data(self, columns, method='onehot'):
        """Encodes categorical dimensions into onehot, ordinal, or uniform numeric representations."""
        if self.df is None or not columns:
            return pd.DataFrame()

        sub_df = self.df[columns].copy()
        if method == 'onehot':
            return pd.get_dummies(sub_df, columns=columns, prefix='onehot', dtype=float)

        elif method == 'ordinal':
            encoded_df = pd.DataFrame(index=self.df.index)
            for col in columns:
                encoded_df[f'ordinal_{col}'] = sub_df[col].astype('category').cat.codes.astype(float)
            return encoded_df

        elif method == 'uniform':
            encoded_df = pd.DataFrame(index=self.df.index)
            for col in columns:
                codes = sub_df[col].astype('category').cat.codes
                c_max = codes.max()
                encoded_df[f'uniform_{col}'] = (codes / c_max).astype(float) if c_max > 0 else 0.0
            return encoded_df

        return pd.DataFrame()

    def merge_features(self, num_cols, num_method, cat_cols, cat_method):
        """Combines original scaled numerical and encoded categorical matrices."""
        df_num = self.extract_normalized_numeric_data(num_cols, method=num_method)
        df_cat = self.extract_normalized_categorical_data(cat_cols, method=cat_method)
        return pd.concat([df_num, df_cat], axis=1)

    def plot_univariate_analysis(self, column):
        """Generates an interactive 3-panel distribution overview for numeric variables."""
        if self.df is None or column not in self.df.columns:
            return

        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            f"{column} Box & Violin Layout", f"{column} Value Trace Index Plot", f"{column} Distribution Histogram"
        ))

        # Panel 1: Violin & Box layout
        fig.add_trace(graph_objects.Violin(x=self.df[column], box_visible=True, meanline_visible=True, name="Distribution"), row=1, col=1)
        # Panel 2: Trend / Value trace index plot
        fig.add_trace(graph_objects.Scatter(y=self.df[column], mode='markers', name="Index Value"), row=2, col=1)
        # Panel 3: Distribution Histogram
        fig.add_trace(graph_objects.Histogram(x=self.df[column], name="Bins"), row=3, col=1)

        fig.update_layout(height=800, title_text=f"Comprehensive Statistical Field Profile: {column}", showlegend=False)
        fig.show()

    def plot_relationship(self, col1, col2):
        """Detects column types to plot relationships with the correct chart type."""
        if self.df is None or col1 not in self.df.columns or col2 not in self.df.columns:
            return

        is_num1 = np.issubdtype(self.df[col1].dtype, np.number)
        is_num2 = np.issubdtype(self.df[col2].dtype, np.number)

        # Case A: Numeric vs Numeric
        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Numeric Correlation: {col1} vs {col2}")
        # Case B: Categorical vs Numeric
        elif (not is_num1 and is_num2) or (is_num1 and not is_num2):
            cx, ny = (col1, col2) if not is_num1 else (col2, col1)
            fig = px.box(self.df, x=cx, y=ny, points="all", title=f"Class Shift Analysis: {cx} vs {ny}")
        # Case C: Categorical vs Categorical
        else:
            counts = self.df.groupby([col1, col2]).size().reset_index(name='Count')
            fig = px.bar(counts, x=col1, y='Count', color=col2, barmode='group', title=f"Cross-Tab Frequency: {col1} vs {col2}")

        fig.show()

    def plot_all_associations_heatmap(self):
        """Computes and visualizes correlations across mixed-type data matrices."""
        if self.df is None: return

        cols = self.df.columns.tolist()
        n = len(cols)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                c1, c2 = cols[i], cols[j]
                is_num1 = np.issubdtype(self.df[c1].dtype, np.number)
                is_num2 = np.issubdtype(self.df[c2].dtype, np.number)

                # Drop pairwise missing records to safely calculate associations
                valid_data = self.df[[c1, c2]].dropna()
                if valid_data.empty:
                    matrix[i, j] = 0.0
                    continue

                v1, v2 = valid_data[c1], valid_data[c2]

                # If either column has insufficient unique values after dropna, correlation is 0.
                if v1.nunique() <= 1 or v2.nunique() <= 1:
                    matrix[i, j] = 0.0
                    continue

                if is_num1 and is_num2:
                    # Numeric-Numeric: Pearson
                    matrix[i, j] = v1.corr(v2, method='pearson') if len(v1) > 1 and v1.std() > 0 and v2.std() > 0 else 0
                elif not is_num1 and not is_num2:
                    # Categorical-Categorical: Cramér's V
                    confusion_matrix = pd.crosstab(v1, v2)
                    n_obs = confusion_matrix.sum().sum()
                    r, k = confusion_matrix.shape

                    if n_obs > 0 and min(r - 1, k - 1) > 0: # Ensure valid degrees of freedom
                        try:
                            chi2 = stats.chi2_contingency(confusion_matrix)[0]
                            matrix[i, j] = np.sqrt(chi2 / (n_obs * min(r - 1, k - 1)))
                        except ValueError: # Catch potential ValueErrors from chi2_contingency for degenerate tables
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0
                else:
                    # Mixed Type (Num-Cat): Eta coefficient derived via ANOVA
                    num_col, cat_col = (v1, v2) if is_num1 else (v2, v1)
                    groups = [group.values for name, group in num_col.groupby(cat_col) if len(group) > 0]

                    # Ensure enough groups and enough data points for ANOVA
                    if len(groups) > 1 and sum(len(g) for g in groups) > len(groups):
                        try:
                            f_val, _ = stats.f_oneway(*groups)
                            n_total = len(num_col)
                            k_groups = len(groups)
                            # Ensure denominator is not zero for Eta calculation
                            if (f_val * (k_groups - 1) + (n_total - k_groups)) > 0:
                                matrix[i, j] = np.sqrt((f_val * (k_groups - 1)) / (f_val * (k_groups - 1) + (n_total - k_groups)))
                            else:
                                matrix[i, j] = 0.0
                        except ValueError: # Catch potential ValueErrors from f_oneway
                            matrix[i, j] = 0.0
                    else:
                        matrix[i, j] = 0.0

        # Safeguard NaN conversions to absolute boundaries
        matrix = np.nan_to_num(matrix)
        fig = px.imshow(matrix, x=cols, y=cols, color_continuous_scale='RdBu_r', zmin=-1.0, zmax=1.0,
                        title="Unified Association Heatmap (Pearson, Cramer's V, Eta)")
        fig.show()
