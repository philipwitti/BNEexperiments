use std::sync::LazyLock;

use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;
use crate::utils::SysRegex;

const SPLIT_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

static RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(SPLIT_PATTERN).unwrap()
});

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct SplitByteLevel;

impl PreTokenizer for SplitByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, normalized| normalized.split(re_ref, SplitDelimiterBehavior::Isolated))?;
        ByteLevel::new(false, true, false).pre_tokenize(pretokenized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::sequence::Sequence;
    use crate::pre_tokenizers::split::{Split, SplitPattern};
    use crate::pre_tokenizers::PreTokenizerWrapper;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn behaves_like_requested_sequence() {
        let split = Split::new(
            SplitPattern::Regex(SPLIT_PATTERN.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap();
        let sequence = Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ]);
        let custom = SplitByteLevel;

        let text = "Hello, WORLD!\nA test 123";
        let mut a = PreTokenizedString::from(text);
        let mut b = PreTokenizedString::from(text);
        custom.pre_tokenize(&mut a).unwrap();
        sequence.pre_tokenize(&mut b).unwrap();

        assert_eq!(
            a.get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            b.get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>()
        );
    }
}
