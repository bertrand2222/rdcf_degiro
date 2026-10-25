import json
import os
import pandas as pd
import numpy as np
from rdcf_degiro.financial_statements import FinancialStatements, Statements

class InconsistentStatementsError(Exception):
    """ Error retrived when Forcast and Histrory statement are not consistant"""

class FinancialForcast(Statements):
    """
    Class computing and containing fincancial forcast
    """
    FINANCIAL_FCST_CODES =  ['NET', 'SAL', 'PRE', 'CPX', 'CPS', 'EBT']
    currency_convertible_cols = FINANCIAL_FCST_CODES

    def __init__(self, statements : FinancialStatements, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.statements = statements

        self._forcasted_ocf_growth : float = None
        self._forcasted_capex_growth : float = None
        self.forcasted_ebitda_growth : float = None
        self.forcasted_ebit_growth : float = None
        self.forcasted_revenue_growth : float = None
        self.y_forcasts : pd.DataFrame = None

    def retrieve_forcasts(self):
        """
        retrive forcated financial data
        """

        self.logger.info(f'{self.name} : retrieves forcast                    ')
        statement_path = os.path.join(self.session_model.output_folder,
                                    f"{self.symbol}_company_forcast.json")
        if self.session_model.update_statements_need(statement_path):
            try:
                estimates_summaries = self.session_model.degiro_api.get_estimates_summaries(
                product_isin= self.isin,
                raw=True,
                )['data']

            except KeyError as e:
                raise KeyError(f'no forcast statement found for isin {self.isin}') from e

            with open(statement_path,
                        "w", 
                        encoding= "utf8") as json_file: 
                json.dump(estimates_summaries, json_file, indent = 4)
        else :
            with open(statement_path, "r", encoding= "utf8") as json_file:
                estimates_summaries = json.load(json_file)

         ### Retrieve annual data
        data = []
        for y in estimates_summaries['annual'] :
            y_dict = {
                      "endDate"  :  pd.Timestamp(year = int(y['year']), month = 12, day = 31)}
            for statement in y['statements']:
                for  item in  statement['items'] :
                    if item['code'] in self.FINANCIAL_FCST_CODES:
                        y_dict[item["code"]] = item["value"]
            data.append(y_dict)

        self.y_forcasts = pd.DataFrame.from_records(
            data).set_index('endDate').sort_index()
        statements_currency = estimates_summaries['currency']

        self.convert_to_price_currency(['y_forcasts'], statements_currency)

        if 'EBT' not in self.y_forcasts:
            # Set forcasted EBITDA equal to focasted EBIT
            self.y_forcasts['EBT'] = self.y_forcasts['EBI']

    def _get_forcasted_growth(self, ls : list[str]):
        """
        Get the fitted growth rate of a forcasted variable
        """
        if self.y_forcasts is None:
            return None
        for val in ls :
            if val not in self.y_forcasts :
                continue
            ys  = self.y_forcasts[val].dropna()
            ys = ys.loc[ys >0 ]
            if len(ys) <= 1 :
                continue
            y = np.log(ys.values / ys.iloc[0])
            x = ys.index.year
            z = np.polyfit(x,y, deg = 1)
            g = np.exp(z[0]) - 1

            return g

        
        self.logger.warning(f"{self.name} no valid value to compute growth estimate from {", ".join(ls)}")
        return None

    @property
    def forcasted_ocf_growth(self):
        """
        forcasted annual growth rate
        """
        if self._forcasted_ocf_growth is None :
            self._forcasted_ocf_growth = self._get_forcasted_growth(['CPS', 'EBT',  'PRE', 'SAL', 'NET'])
        return self._forcasted_ocf_growth


    def _set_forcasted_ocf(self):
        """
        retruned forcasted ocf array
        """
        self._forcasted_ocfe = None
        if self.y_forcasts is None:
            return
        net_interest_exp = 0
        last_ocfe = self.statements.y_statements['OCFE'].iloc[-1]

        if 'SNIN' in self.statements.y_statements:
            # interest expense assumed constant
            net_interest_exp = self.statements.interest_expense_income * (1 - self.session_model.taxe_rate)

        if 'CPS'  in self.y_forcasts:
            ratio = last_ocfe/self.y_forcasts['CPS'].iloc[0]
            # compute forcasted ocf from forcasted cash flow per share
            self._forcasted_ocfe = self.y_forcasts['CPS'].dropna() * ratio
            self._forcasted_ocff = self._forcasted_ocfe - net_interest_exp

        elif ('EBT' in self.y_forcasts) and ('EBI' in self.y_forcasts) :
            # compute forcasted ocf from forcasted ebitda and ebit
            self._forcasted_ocff = (self.y_forcasts['EBT'] - self.session_model.taxe_rate * self.y_forcasts['EBI']).dropna()
            last_ocff = self.statements.y_statements['OCFF'].iloc[-1]
            self._forcasted_ocff *= last_ocff/self._forcasted_ocff.iloc[0]
            self._forcasted_ocfe = self._forcasted_ocff + net_interest_exp
        else:
            # compute forcasted ocf from forcasted ebitda, ebt, net income, pretax income or sales growth
            for val in ['EBT', 'EBI', 'NET', 'PRE', 'SAL' ] :
                if val not in self.y_forcasts:
                    continue
                ys = self.y_forcasts[val].dropna()

                # rescale variable array to ratio between last stated ocf 
                # and first variable value  
                ratio = last_ocfe/ys.iloc[0]
                if (ys.index[0].year == self.statements.y_statements.index[-1].year) and ratio > 0:
                    ys *= ratio
                    self._forcasted_ocfe = ys
            
        if self._forcasted_ocfe is None:
            self.logger.warning(f"{self.name} Forcasted cash flow could not be computed from forcast data")
            self._forcasted_ocfe = self.statements.ocfe * (1 + self.forcasted_ocf_growth)**np.arange(1,1 +self.session_model.nb_year_dcf)
            self._forcasted_ocff = self._forcasted_ocfe - net_interest_exp
            return

        # Set forcast array lenght equal to dcf model nb year
        if len(self._forcasted_ocfe) >= self.session_model.nb_year_dcf:
            self._forcasted_ocfe = self._forcasted_ocfe.iloc[:self.session_model.nb_year_dcf]
        else :
            # complete forcasted ocf array with value extrapolated from forcasted growth rate
            self._forcasted_ocfe = np.concat([
                    self._forcasted_ocfe,
                    self._forcasted_ocfe.iloc[-1] * (1+ self.forcasted_ocf_growth)**np.arange(
                        1,
                        1 + self.session_model.nb_year_dcf - len(self._forcasted_ocfe))])
        
        self._forcasted_ocff = self._forcasted_ocfe - net_interest_exp

    def _get_forcasted_ebidta(self):
        """
        retruned forcasted ocf array
        """
        if self.y_forcasts is None:
            return None

        f_year = self.y_forcasts.index[0].year 
        h_year = self.statements.y_statements.index[-1].year
        if h_year !=  f_year :
            raise InconsistentStatementsError(f"Inconsistent first forcast year and last history report year: {f_year} {h_year}" )
        revenu_ratio = self.statements.y_statements['RTLR'].iloc[-1] / self.y_forcasts['SAL'].iloc[0]
        if np.isinf(revenu_ratio):
            raise InconsistentStatementsError(f"Inconsistent first forcast year and last history report year revenu: {self.statements.y_statements['RTLR'].iloc[-1]} {self.y_forcasts['SAL'].iloc[0]}")
        ys = None

        for val in ['EBT', 'EBI',] :
            if val in self.y_forcasts:
                ys = self.y_forcasts[val] * revenu_ratio

                if ys.isna().iloc[0]:
                    ys.iloc[0] = self.statements.y_statements['EBITDA'].iloc[-1]
                
                ys = ys.dropna()
                # rescale data  to ratio between last stated ebitda and first array value
                # f_year = ys.index[0].year 
                # h_year = self.statements.y_statements.index[-1].year
                # ratio = self.statements.y_statements['EBITDA'].iloc[-1]/ys.iloc[0]
                # if (ys.index[0].year != h_year) or ratio < 0:
                #     raise InconsistentStatementsError(f"Inconsistent EBITDA stated in forcast and history report: year {f_year} {h_year} ratio :{ratio}")
                # ys *=ratio
                
                if len(ys) >= self.session_model.nb_year_dcf:
                    return ys[:self.session_model.nb_year_dcf]

                if self.forcasted_ebitda_growth is None :
                    continue
                
                # complete forcasted array with value extrapolated from forcasted growth rate
                ys = np.concat([
                        ys,
                        ys.iloc[-1] * (1+ self.forcasted_ebitda_growth)**np.arange(
                            1,
                            1 + self.session_model.nb_year_dcf - len(ys))])
            
                return ys

        # no forcasted cash flow per share provided
        return self.statements.ebitda * (1 + self.forcasted_ocf_growth)**np.arange(1,1 +self.session_model.nb_year_dcf)

    def _set_forcasted_capex(self):
        """
        retruned capital expenditure growth rate fited from estimate 
        """
        if self.y_forcasts is None:
            return

        if 'CPX' in self.y_forcasts:
            ys = self.y_forcasts['CPX'].dropna()
            if len(ys) >= self.session_model.nb_year_dcf:
                self._forcasted_capex = ys[:self.session_model.nb_year_dcf]
                return
            
            g = self._get_forcasted_growth(['CPX'])
            if not np.isnan(g) :
                g = min(g,2) # bound growth rate to 2
                self._forcasted_capex_growth = g 
                ys = np.concat([
                        ys,
                        ys.iloc[-1] * (1+ g)**np.arange(
                            1,
                            1 + self.session_model.nb_year_dcf - len(ys))])
                self._forcasted_capex = ys
                return

        # can not compute forcasted capital expenditure growth from itself
        self._forcasted_capex_growth = self.forcasted_ocf_growth
        self._forcasted_capex = self.statements.capex * (1 + self._forcasted_capex_growth)**np.arange(1,1 +self.session_model.nb_year_dcf)

    def compute_forcasted_data(self):

        self._set_forcasted_ocf()
        self._set_forcasted_capex()
        if self._forcasted_ocfe is None:
            return
        self.forcasted_fcfe = self._forcasted_ocfe - self._forcasted_capex
        self.forcasted_fcff = self._forcasted_ocff - self._forcasted_capex
        self.forcasted_ebitda_growth = self._get_forcasted_growth(['EBT', 'EBI'])
