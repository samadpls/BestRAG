use pyo3::prelude::*;

#[pymodule]
fn _bestrag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", crate::VERSION)?;
    Ok(())
}
