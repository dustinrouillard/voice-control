/// Fixed-capacity ring of the most recent samples.
///
/// "computa, mute" is spoken in one breath, so by the time the wake
/// word scores a detection the command is already partly in the past.
/// The pipeline seeds its capture buffer from here so those samples
/// are not lost.
pub struct PreRoll {
  buf: Vec<f32>,
  head: usize,
  filled: usize,
}

impl PreRoll {
  pub fn new(capacity: usize) -> Self {
    Self {
      buf: vec![0.0; capacity.max(1)],
      head: 0,
      filled: 0,
    }
  }

  pub fn extend(&mut self, samples: &[f32]) {
    let cap = self.buf.len();

    // Anything older than the last `cap` samples is about to be
    // overwritten anyway, so skip straight to the part that survives.
    let start = samples.len().saturating_sub(cap);

    for &sample in &samples[start..] {
      self.buf[self.head] = sample;
      self.head = (self.head + 1) % cap;
      self.filled = (self.filled + 1).min(cap);
    }
  }

  /// The most recent `n` samples, oldest first. Returns fewer than
  /// `n` if the ring has not seen that many yet.
  pub fn tail(&self, n: usize) -> Vec<f32> {
    let cap = self.buf.len();
    let take = n.min(self.filled);
    let mut out = Vec::with_capacity(take);

    for i in 0..take {
      // `head` points one past the newest sample.
      let idx = (self.head + cap - take + i) % cap;
      out.push(self.buf[idx]);
    }

    out
  }

  pub fn clear(&mut self) {
    self.head = 0;
    self.filled = 0;
  }
}

#[cfg(test)]
mod tests {
  use super::PreRoll;

  #[test]
  fn tail_returns_most_recent_samples_oldest_first() {
    let mut ring = PreRoll::new(4);
    ring.extend(&[1.0, 2.0, 3.0]);

    assert_eq!(ring.tail(2), vec![2.0, 3.0]);
    assert_eq!(ring.tail(9), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn wraps_and_keeps_only_the_newest() {
    let mut ring = PreRoll::new(3);
    ring.extend(&[1.0, 2.0]);
    ring.extend(&[3.0, 4.0, 5.0]);

    assert_eq!(ring.tail(3), vec![3.0, 4.0, 5.0]);
  }

  #[test]
  fn a_single_write_larger_than_capacity_keeps_the_newest() {
    let mut ring = PreRoll::new(3);
    ring.extend(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_eq!(ring.tail(3), vec![3.0, 4.0, 5.0]);
  }
}
