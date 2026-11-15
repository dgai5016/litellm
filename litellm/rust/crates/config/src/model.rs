use std::fmt;

use litellm_router_types::LitellmParams;
use serde::Deserialize;

use crate::{AdditionalFields, Object, Value};

#[derive(Clone, Deserialize)]
#[serde(try_from = "Written")]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LitellmParams,
    pub written_params: Value,
    pub model_info: Object,
    pub blocked: Option<bool>,
    pub additional_fields: AdditionalFields,
}

#[derive(Deserialize)]
struct Written {
    model_name: String,
    litellm_params: Value,
    #[serde(default)]
    model_info: Object,
    blocked: Option<bool>,
    #[serde(flatten)]
    additional_fields: AdditionalFields,
}

impl TryFrom<Written> for Model {
    type Error = serde_yaml_ng::Error;

    fn try_from(written: Written) -> Result<Self, Self::Error> {
        Ok(Self {
            model_name: written.model_name,
            litellm_params: serde_yaml_ng::from_value(written.litellm_params.clone())?,
            written_params: written.litellm_params,
            model_info: written.model_info,
            blocked: written.blocked,
            additional_fields: written.additional_fields,
        })
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("model_name", &self.model_name)
            .field("litellm_params", &self.litellm_params)
            .field("model_info", &self.model_info)
            .field("blocked", &self.blocked)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}
