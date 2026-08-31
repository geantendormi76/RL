use parking_lot::Mutex;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum VramError {
    #[error("显存预算溢出: 申请 {requested} MB, 当前已用 {current} MB, 硬上限 {limit} MB")]
    OutOfVramBudget {
        requested: usize,
        current: usize,
        limit: usize,
    },
}

#[derive(Debug)]
struct VramState {
    allocated_mb: usize,
    hard_limit_mb: usize,
}

#[derive(Clone, Debug)]
pub struct VramManager {
    state: Arc<Mutex<VramState>>,
}

impl VramManager {
    pub const DEFAULT_HARD_LIMIT_MB: usize = 11_000;

    pub fn new(hard_limit_mb: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(VramState {
                allocated_mb: 0,
                hard_limit_mb,
            })),
        }
    }

    pub fn default_rtx3060() -> Self {
        Self::new(Self::DEFAULT_HARD_LIMIT_MB)
    }

    pub fn acquire(&self, mb: usize) -> Result<VramTokenGuard, VramError> {
        let mut state = self.state.lock();
        if state.allocated_mb + mb > state.hard_limit_mb {
            return Err(VramError::OutOfVramBudget {
                requested: mb,
                current: state.allocated_mb,
                limit: state.hard_limit_mb,
            });
        }
        state.allocated_mb += mb;
        Ok(VramTokenGuard {
            allocated_mb: mb,
            state: Arc::clone(&self.state),
        })
    }

    pub fn current_usage_mb(&self) -> usize {
        self.state.lock().allocated_mb
    }

    pub fn available_mb(&self) -> usize {
        let state = self.state.lock();
        state.hard_limit_mb.saturating_sub(state.allocated_mb)
    }
}

#[derive(Debug)]
pub struct VramTokenGuard {
    allocated_mb: usize,
    state: Arc<Mutex<VramState>>,
}

impl VramTokenGuard {
    pub fn mb(&self) -> usize {
        self.allocated_mb
    }
}

impl Drop for VramTokenGuard {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.allocated_mb = state.allocated_mb.saturating_sub(self.allocated_mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_acquire_and_auto_drop() {
        let mgr = VramManager::new(11_000);
        assert_eq!(mgr.current_usage_mb(), 0);
        assert_eq!(mgr.available_mb(), 11_000);

        {
            let guard1 = mgr.acquire(3_500).expect("应该成功分配 3.5GB 静态模型");
            assert_eq!(mgr.current_usage_mb(), 3_500);
            assert_eq!(guard1.mb(), 3_500);

            {
                let guard2 = mgr.acquire(2_000).expect("应该成功分配 2.0GB KV Cache");
                assert_eq!(mgr.current_usage_mb(), 5_500);
                assert_eq!(guard2.mb(), 2_000);
            } // guard2 离开作用域，自动归还 2000 MB

            assert_eq!(mgr.current_usage_mb(), 3_500);
        } // guard1 离开作用域，自动归还 3500 MB

        assert_eq!(mgr.current_usage_mb(), 0);
    }

    #[test]
    fn test_vram_out_of_budget_rejection() {
        let mgr = VramManager::new(11_000);
        let _guard = mgr.acquire(10_000).expect("成功分配 10GB");

        let err = mgr.acquire(1_500).unwrap_err();
        assert_eq!(
            err,
            VramError::OutOfVramBudget {
                requested: 1_500,
                current: 10_000,
                limit: 11_000
            }
        );
    }
}
