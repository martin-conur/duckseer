use augurs::ets::trend::AutoETSTrendModel;
use augurs::{
    ets::AutoETS,
    forecaster::{
        transforms::{LinearInterpolator, MinMaxScaler},
        Forecaster,
        Transformer,
    },
    mstl::MSTLModel,
    Forecast,
    Fit,
    Predict
};
use anyhow::{Context, Result, bail};

#[derive(Copy, Clone, Debug)]
pub enum InterpolationMethod {
    Linear,
    Seasonal
}

#[derive(Copy, Clone, Debug)]
pub enum ForecastMethod {
    MSTL, 
    ETS,
    Prophet
}

#[derive(Clone, Debug)]
pub struct GuardrailResult {
    pub safe_forecast: Vec<f64>,
    pub is_modified: Vec<bool>
}

impl GuardrailResult {
    pub fn new(safe_forecast: Vec<f64>, is_modified: Vec<bool>) -> Self {
        Self {
            safe_forecast,
            is_modified
        }
    }

    pub fn clamp_to_bounds(value: f64, lower: f64, upper: f64) -> (f64, bool) {
        if value >= lower && value <= upper {
            (value, false)
        } else if value < lower {
            (lower, true)
        } else {
            (upper, true)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum GuardrailTightness {
    Loose,
    Medium,
    Tight
}

impl GuardrailTightness {
    pub fn tightness_to_confidence(self) -> f64 {
        match self {
            Self::Loose => 0.95,
            Self::Medium => 0.65,
            Self::Tight => 0.4,
        }
    }
}

const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

pub fn get_default_periods(data: &[f64]) -> Result<Vec<usize>> {
    match data.len() {
        0..4 => bail!("Not enough data points!"),
        _ => {
            Ok(vec![data.len() / 2])
        }
    }
}

pub fn create_mstl_forecaster(periods: Vec<usize>) -> Result<Forecaster<MSTLModel<AutoETSTrendModel>>> {
    let ets = AutoETS::non_seasonal().into_trend_model();
    let mstl = MSTLModel::new(periods, ets);

    let transformers = vec![
        LinearInterpolator::new().boxed(),
        MinMaxScaler::new().boxed(),
    ];

    let forecaster = Forecaster::new(mstl).with_transformers(transformers);
    Ok(forecaster)
}

pub fn fill_missing_data(data: &[f64], method: InterpolationMethod, periods: Option<Vec<usize>>) -> Result<Vec<f64>> {
    let periods = periods.unwrap_or(get_default_periods(data)?);

    match method {
        InterpolationMethod::Linear => {
            let mut interpolated = data.to_vec();
            let _ = LinearInterpolator::new().transform(&mut interpolated);
            Ok(interpolated)
        },
        InterpolationMethod::Seasonal => {
            let mut forecaster = create_mstl_forecaster(periods)?;

            forecaster.fit(data).context("Failed to fit forecaster model")?;

            let in_sample = forecaster
                .predict_in_sample(DEFAULT_CONFIDENCE_LEVEL)
                .context("Failed to generate in-sample predictions")?;

            let result = data.iter().enumerate().map(|(i, &v)| {
                if v.is_nan() {
                    in_sample.point[i]
                } else {
                    v
                }
            }).collect();
            Ok(result)
        }
    }
}

pub fn mstl_forecast(data: &[f64], periods: Option<Vec<usize>>, horizon: usize) -> Result<Forecast> {
    let periods = periods.unwrap_or(get_default_periods(data)?);
    let mut forecaster = create_mstl_forecaster(periods)?;
    forecaster.fit(data)?;
    let forecast_results = forecaster.predict(horizon, DEFAULT_CONFIDENCE_LEVEL)?;
    Ok(forecast_results)
}

pub fn ets_forecast(data: &[f64], horizon: usize, confidence_interval: f64) -> Result<Forecast> {
    let model = AutoETS::new(data.len().min(10), "ZAZ")?; // error + trend + seasonality
    let search = model.fit(data)?;
    let forecast_results = search.predict(horizon, confidence_interval)?;
    Ok(forecast_results)
}

pub fn guardrails(data: &[f64], forecasted: &[f64], method: ForecastMethod, tightness: GuardrailTightness) -> Result<GuardrailResult> {
    let horizon = forecasted.len();
    let confidence_interval = tightness.tightness_to_confidence();

    let forecast_with_intervals = match method {
        ForecastMethod::ETS => {
            ets_forecast(data, horizon, confidence_interval)?
        },
        ForecastMethod::MSTL => mstl_forecast(data, None, horizon)?,
        ForecastMethod::Prophet => bail!("Prophet model not yet implemented"),
    };

    if let Some(interval) = forecast_with_intervals.intervals {
        let lower_bound = interval.lower;
        let upper_bound = interval.upper;
        let mut safe_forecast = Vec::with_capacity(horizon);
        let mut is_modified = Vec::with_capacity(horizon);

        for ((&upper, &lower), &forecasted) in upper_bound.iter()
            .zip(lower_bound.iter()).zip(forecasted.iter())
        {
            let (value, modified) = GuardrailResult::clamp_to_bounds(forecasted, lower, upper);
            safe_forecast.push(value);
            is_modified.push(modified);
        }

        Ok(GuardrailResult::new(safe_forecast, is_modified))
    } else {
        bail!("Forecast didn't produce confidence intervals")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ets_forecast() {
        let data = [85.0, 92.0, 78.0, 75.0, 73.0, 74.0, 72.0, 68.0, 69.0, 67.0];
        let result = ets_forecast(&data, 4, 0.95);
        assert!(result.is_ok());
    }

    #[test]
    fn test_guardrails() {
        let data = [85.0, 92.0, 78.0, 75.0, 73.0, 74.0, 72.0, 68.0, 69.0, 67.0];
        let forecasted = [65.0, 80.0, 150.0, 50.0];
        let result = guardrails(&data, &forecasted, ForecastMethod::ETS, GuardrailTightness::Loose);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clamp_to_bounds() {
        let (value, modified) = GuardrailResult::clamp_to_bounds(150.0, 60.0, 100.0);
        assert_eq!(value, 100.0);
        assert!(modified);

        let (value, modified) = GuardrailResult::clamp_to_bounds(80.0, 60.0, 100.0);
        assert_eq!(value, 80.0);
        assert!(!modified);
    }
}
