#[allow(dead_code)]
pub enum LogType {
    Print,
    Println,
    Info,
    Debug,
    Error,
    Trace,
}

/// Prints out the provided message through a logger controlled by the logtype param.
///
/// # Example
///
/// ```no run
/// # fn main {
///     show_message!("Hello!", LogType::Info);
/// #}
#[macro_export]
macro_rules! show_message {
    ($msg: expr, $logtype: expr) => {
        use crate::macros::LogType;
        use log::{debug, error, info, trace};
        match $logtype {
            LogType::Print => print!("{}", $msg),
            LogType::Println => println!("{}", $msg),
            LogType::Info => info!("{}", $msg),
            LogType::Debug => debug!("{}", $msg),
            LogType::Error => error!("{}", $msg),
            LogType::Trace => trace!("{}", $msg),
        }
    };
}

/// Measures the time duration of the provided function and prints out the message
/// through a logger controlled by the logtype param.
/// The $msg param exists to specify an unique identifier text, so that it's easier
/// to track the results.
///
/// # Example
///
/// ```no run
/// # fn main {
///     measure_time!("testing measure", || { println!("This is some func") }, LogType::Info);
/// #}
#[macro_export]
macro_rules! measure_time {
    ($msg: expr, $fn: expr) => {{
        let instant = std::time::Instant::now();
        let res = $fn();
        debug!(
            "Finished \"{}\" in {} ms",
            $msg,
            instant.elapsed().as_millis(),
        );
        res
    }};
    ($msg: expr, $fn: expr, $logtype: expr) => {{
        let instant = std::time::Instant::now();
        let res = $fn();
        $crate::show_message!(
            format!(
                "Finished \"{}\" in {} ms",
                $msg,
                instant.elapsed().as_millis(),
            ),
            $logtype
        );
        res
    }};
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    #[test]
    fn measure_time_no_return_value() {
        let mut val = 0;
        assert_eq!(
            measure_time!(
                "testing measure",
                || {
                    val += 1;
                },
                LogType::Info
            ),
            ()
        );
        assert_eq!(val, 1);
    }

    #[test]
    fn measure_time_with_return_value() {
        let val = 0;
        assert_eq!(
            measure_time!("testing measure", || { val + 1 }, LogType::Info),
            1
        );
    }

    #[test]
    fn measure_time_ok_result() -> Result<()> {
        use image::{open, DynamicImage};
        let res = measure_time!(
            "loading values",
            || -> Result<DynamicImage> {
                let image = open("./test_data/polygon.png")?;
                Ok(image)
            },
            LogType::Error
        );
        assert!(res.is_ok());
        let image = res.unwrap().to_luma();
        assert_eq!(image.width(), 300);
        assert_eq!(image.height(), 300);
        Ok(())
    }

    #[test]
    fn measure_time_err_result() -> Result<()> {
        use image::{open, DynamicImage};
        let res = measure_time!(
            "loading values",
            || -> Result<DynamicImage> {
                let image = open("./test_data/non-existing.png")?;
                Ok(image)
            },
            LogType::Error
        );
        assert!(res.is_err());
        Ok(())
    }
}
