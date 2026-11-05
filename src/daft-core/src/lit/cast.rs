use common_error::DaftResult;
use half::f16;
use num_traits::cast::cast as num_cast;

use super::Literal;
use crate::datatypes::DataType;

/// Direct scalar cast rules that mirror the Arrow casts used by `Series::cast`.
///
/// Returns `None` when the source/target pair has no direct rule yet; the
/// caller then falls back to casting through a one-element `Series`.
pub(super) fn cast_literal(lit: &Literal, dtype: &DataType) -> Option<DaftResult<Literal>> {
    if dtype.is_null() {
        return Some(Ok(Literal::Null));
    }
    if matches!(lit, Literal::Null) {
        return Some(Ok(Literal::Null));
    }

    match lit {
        Literal::Boolean(value) if is_physical_numeric(dtype) => {
            Some(Ok(bool_to_numeric(*value, dtype)))
        }
        _ if is_physical_numeric(dtype) && is_physical_numeric_lit(lit) => {
            Some(Ok(numeric_to_numeric(lit, dtype)))
        }
        _ if matches!(dtype, DataType::Boolean) && is_physical_numeric_lit(lit) => {
            Some(Ok(numeric_to_bool(lit)))
        }
        _ => None,
    }
}

fn is_physical_numeric(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn is_physical_numeric_lit(lit: &Literal) -> bool {
    matches!(
        lit,
        Literal::Int8(_)
            | Literal::Int16(_)
            | Literal::Int32(_)
            | Literal::Int64(_)
            | Literal::UInt8(_)
            | Literal::UInt16(_)
            | Literal::UInt32(_)
            | Literal::UInt64(_)
            | Literal::Float16(_)
            | Literal::Float32(_)
            | Literal::Float64(_)
    )
}

/// Matches Arrow's numeric cast semantics: values that do not fit the target
/// type become null instead of wrapping or erroring.
fn numeric_to_numeric(lit: &Literal, dtype: &DataType) -> Literal {
    macro_rules! cast {
        ($value:expr) => {
            match dtype {
                DataType::Int8 => num_cast::<_, i8>($value)
                    .map(Literal::Int8)
                    .unwrap_or(Literal::Null),
                DataType::Int16 => num_cast::<_, i16>($value)
                    .map(Literal::Int16)
                    .unwrap_or(Literal::Null),
                DataType::Int32 => num_cast::<_, i32>($value)
                    .map(Literal::Int32)
                    .unwrap_or(Literal::Null),
                DataType::Int64 => num_cast::<_, i64>($value)
                    .map(Literal::Int64)
                    .unwrap_or(Literal::Null),
                DataType::UInt8 => num_cast::<_, u8>($value)
                    .map(Literal::UInt8)
                    .unwrap_or(Literal::Null),
                DataType::UInt16 => num_cast::<_, u16>($value)
                    .map(Literal::UInt16)
                    .unwrap_or(Literal::Null),
                DataType::UInt32 => num_cast::<_, u32>($value)
                    .map(Literal::UInt32)
                    .unwrap_or(Literal::Null),
                DataType::UInt64 => num_cast::<_, u64>($value)
                    .map(Literal::UInt64)
                    .unwrap_or(Literal::Null),
                DataType::Float16 => num_cast::<_, f16>($value)
                    .map(Literal::Float16)
                    .unwrap_or(Literal::Null),
                DataType::Float32 => num_cast::<_, f32>($value)
                    .map(Literal::Float32)
                    .unwrap_or(Literal::Null),
                DataType::Float64 => num_cast::<_, f64>($value)
                    .map(Literal::Float64)
                    .unwrap_or(Literal::Null),
                _ => unreachable!("checked by is_physical_numeric"),
            }
        };
    }

    match lit {
        Literal::Int8(value) => cast!(*value),
        Literal::Int16(value) => cast!(*value),
        Literal::Int32(value) => cast!(*value),
        Literal::Int64(value) => cast!(*value),
        Literal::UInt8(value) => cast!(*value),
        Literal::UInt16(value) => cast!(*value),
        Literal::UInt32(value) => cast!(*value),
        Literal::UInt64(value) => cast!(*value),
        Literal::Float16(value) => cast!(*value),
        Literal::Float32(value) => cast!(*value),
        Literal::Float64(value) => cast!(*value),
        _ => unreachable!("checked by is_physical_numeric_lit"),
    }
}

fn bool_to_numeric(value: bool, dtype: &DataType) -> Literal {
    numeric_to_numeric(&Literal::Int8(i8::from(value)), dtype)
}

/// Matches Arrow's `cast_num_to_bool`: zero is false and any other value,
/// including NaN, is true.
fn numeric_to_bool(lit: &Literal) -> Literal {
    let value = match lit {
        Literal::Int8(value) => *value != 0,
        Literal::Int16(value) => *value != 0,
        Literal::Int32(value) => *value != 0,
        Literal::Int64(value) => *value != 0,
        Literal::UInt8(value) => *value != 0,
        Literal::UInt16(value) => *value != 0,
        Literal::UInt32(value) => *value != 0,
        Literal::UInt64(value) => *value != 0,
        Literal::Float16(value) => *value != f16::ZERO,
        Literal::Float32(value) => *value != 0.0,
        Literal::Float64(value) => *value != 0.0,
        _ => unreachable!("checked by is_physical_numeric_lit"),
    };
    Literal::Boolean(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Series;

    fn series_cast(lit: &Literal, dtype: &DataType) -> Literal {
        Series::from(lit.clone())
            .cast(dtype)
            .and_then(|s| Literal::try_from_single_value_series(&s))
            .unwrap()
    }

    fn assert_direct_and_parity(lit: Literal, dtype: DataType) {
        let direct = cast_literal(&lit, &dtype)
            .expect("expected a direct cast rule")
            .unwrap();
        let expected = series_cast(&lit, &dtype);
        assert_eq!(direct, expected, "casting {lit:?} to {dtype:?}");
        assert_eq!(lit.cast(&dtype).unwrap(), expected);
    }

    #[test]
    fn test_direct_numeric_casts() {
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
        ] {
            assert_direct_and_parity(Literal::Int8(42), dtype.clone());
            assert_direct_and_parity(Literal::UInt64(42), dtype.clone());
            assert_direct_and_parity(Literal::Float64(1.5), dtype.clone());
            assert_direct_and_parity(Literal::Float16(f16::from_f32(-2.5)), dtype);
        }
    }

    #[test]
    fn test_direct_numeric_casts_null_on_overflow() {
        assert_direct_and_parity(Literal::Int64(300), DataType::Int8);
        assert_direct_and_parity(Literal::Int64(-1), DataType::UInt8);
        assert_direct_and_parity(Literal::UInt64(u64::MAX), DataType::Int64);
        assert_direct_and_parity(Literal::Float64(f64::INFINITY), DataType::Int32);
        assert_direct_and_parity(Literal::Float64(f64::NAN), DataType::Int64);
    }

    #[test]
    fn test_direct_bool_casts() {
        assert_direct_and_parity(Literal::Boolean(true), DataType::Int8);
        assert_direct_and_parity(Literal::Boolean(false), DataType::UInt64);
        assert_direct_and_parity(Literal::Boolean(true), DataType::Float64);
    }

    #[test]
    fn test_direct_numeric_to_bool() {
        for value in [
            Literal::Int8(0),
            Literal::Int64(7),
            Literal::UInt32(1),
            Literal::Float16(f16::ZERO),
            Literal::Float32(0.0),
            Literal::Float64(-0.0),
            Literal::Float64(0.5),
            Literal::Float64(f64::NAN),
        ] {
            assert_direct_and_parity(value, DataType::Boolean);
        }
    }

    #[test]
    fn test_null_casts_are_direct() {
        let casted = cast_literal(&Literal::Null, &DataType::Int32)
            .expect("null casts should not fall back to a Series")
            .unwrap();
        assert_eq!(casted, Literal::Null);
        assert_eq!(
            Literal::Int32(1).cast(&DataType::Null).unwrap(),
            Literal::Null
        );
    }

    #[test]
    fn test_unsupported_casts_fall_back() {
        assert!(cast_literal(&Literal::Decimal(100, 10, 2), &DataType::Int32).is_none());
        assert!(cast_literal(&Literal::Utf8("1".to_string()), &DataType::Int32).is_none());
        assert!(cast_literal(&Literal::Int32(1), &DataType::Utf8).is_none());
        assert!(cast_literal(&Literal::Date(1), &DataType::Int32).is_none());
    }
}
