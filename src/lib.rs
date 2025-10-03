pub mod fft_simd;
mod index_generator;
use crate::index_generator::IndexGen;
use wasm_bindgen::prelude::*;

/// Create a lookup table for faster FFT computation
#[wasm_bindgen]
pub fn generate_lookup_table(length: usize) -> Vec<f32> {
  let length_f32 = 2.0 * length as f32;
  (0..(length << 1))
    .map(|c| {
      if c & 1 == 0 {
        ((c as f32) * (std::f32::consts::PI * 2.0) / length_f32).cos()
      } else {
        -(((c - 1) as f32) * (std::f32::consts::PI * 2.0) / length_f32).sin()
      }
    })
    .collect::<Vec<f32>>()
}

/// Generalize process of radix-4 merge
/// for block size of n and 2*n where n -> 2^k, k \in Natural numbers
fn radx4fft_merge_n_2n(
  out: &mut [f32],
  lookup_table: &[f32],
  block_size: usize,
  length_lookup_incr: usize,
) {
  // Chunks of block_size
  let block_jump = block_size << 1;
  let block_jump_radx4 = block_jump << 1;
  let lookup_incr = length_lookup_incr << 1;
  let lookup_incr_radx4 = length_lookup_incr;
  let length = out.len() >> 1;
  let mid = length >> 1;

  out.chunks_mut(block_jump_radx4).for_each(|out_chunk| {
    let (mut wr, mut wi) = (1.0, 0.0);
    let (mut w2r, mut w2i) = (1.0, 0.0);
    let (mut w3r, mut w3i) = (lookup_table[mid], -lookup_table[mid + 1]);
    let mut lookup_index = 0;
    let mut lookup_index_radx4 = 0;

    for index in (0..block_size).step_by(2) {
      // First Block
      let (c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i) = (
        out_chunk[index],
        out_chunk[index + 1],
        out_chunk[index + block_size],
        out_chunk[index + block_size + 1],
        out_chunk[index + block_jump],
        out_chunk[index + block_jump + 1],
        out_chunk[index + block_size + block_jump],
        out_chunk[index + block_size + block_jump + 1],
      );

      let (c2r, c2i, c4r, c4i) = (
        c2r * wr - c2i * wi,
        c2r * wi + c2i * wr,
        c4r * wr - c4i * wi,
        c4r * wi + c4i * wr,
      );
      let (i1r, i1i, i2r, i2i, i3r, i3i, i4r, i4i) = (
        c1r + c2r,
        c1i + c2i,
        c1r - c2r,
        c1i - c2i,
        c3r + c4r,
        c3i + c4i,
        c3r - c4r,
        c3i - c4i,
      );

      let (i3r, i3i, i4r, i4i) = (
        i3r * w2r - i3i * w2i,
        i3r * w2i + i3i * w2r,
        i4r * w3r - i4i * w3i,
        i4r * w3i + i4i * w3r,
      );

      out_chunk[index] = i1r + i3r;
      out_chunk[index + 1] = i1i + i3i;

      out_chunk[index + block_size] = i2r + i4r;
      out_chunk[index + block_size + 1] = i2i + i4i;

      out_chunk[index + block_jump] = i1r - i3r;
      out_chunk[index + block_jump + 1] = i1i - i3i;

      out_chunk[index + block_jump + block_size] = i2r - i4r;
      out_chunk[index + block_jump + block_size + 1] = i2i - i4i;

      lookup_index += lookup_incr;
      lookup_index_radx4 += lookup_incr_radx4;

      (wr, wi) = (lookup_table[lookup_index], -lookup_table[lookup_index + 1]);
      (w2r, w2i) = (
        lookup_table[lookup_index_radx4],
        -lookup_table[lookup_index_radx4 + 1],
      );
      (w3r, w3i) = (
        lookup_table[lookup_index_radx4 + mid],
        -lookup_table[lookup_index_radx4 + mid + 1],
      );
    }
  });
}

fn radx4fft_merge_n(
  out: &mut [f32],
  lookup_table: &[f32],
  block_size: usize,
  length_check_lookup: usize,
) {
  let lookup_incr = length_check_lookup << 1;
  let block_jump = block_size << 1;

  out.chunks_mut(block_jump).for_each(|out_chunk| {
    let (mut wr, mut wi) = (1.0, 0.0);
    let mut lookup_index = 0;

    for index in (0..block_size).step_by(2) {
      let (c1r, c1i) = (out_chunk[index], out_chunk[index + 1]);
      let (mut c2r, mut c2i) = (
        out_chunk[index + block_size],
        out_chunk[index + block_size + 1],
      );
      let temp = c2r * wr - c2i * wi;
      c2i = c2r * wi + c2i * wr;
      c2r = temp;

      out_chunk[index] = c1r + c2r;
      out_chunk[index + 1] = c1i + c2i;
      out_chunk[index + block_size] = c1r - c2r;
      out_chunk[index + block_size + 1] = c1i - c2i;

      lookup_index += lookup_incr;
      wr = lookup_table[lookup_index];
      wi = -lookup_table[lookup_index + 1];
    }
  });
}

/// Generalize process of radix-4 merge
/// for block size of n and 2*n where n -> 2^k, k \in Natural numbers
fn radx4ifft_merge_n_2n(
  out: &mut [f32],
  lookup_table: &[f32],
  block_size: usize,
  length_lookup_incr: usize,
) {
  // Chunks of block_size
  let block_jump = block_size << 1;
  let block_jump_radx4 = block_jump << 1;
  let lookup_incr = length_lookup_incr << 1;
  let lookup_incr_radx4 = length_lookup_incr;
  let length = out.len() >> 1;
  let mid = length >> 1;

  out.chunks_mut(block_jump_radx4).for_each(|out_chunk| {
    let (mut wr, mut wi) = (1.0, 0.0);
    let (mut w2r, mut w2i) = (1.0, 0.0);
    let (mut w3r, mut w3i) = (lookup_table[mid], lookup_table[mid + 1]);
    let mut lookup_index = 0;
    let mut lookup_index_radx4 = 0;

    for index in (0..block_size).step_by(2) {
      // First Block
      let (c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i) = (
        out_chunk[index],
        out_chunk[index + 1],
        out_chunk[index + block_size],
        out_chunk[index + block_size + 1],
        out_chunk[index + block_jump],
        out_chunk[index + block_jump + 1],
        out_chunk[index + block_size + block_jump],
        out_chunk[index + block_size + block_jump + 1],
      );

      let (c2r, c2i, c4r, c4i) = (
        c2r * wr - c2i * wi,
        c2r * wi + c2i * wr,
        c4r * wr - c4i * wi,
        c4r * wi + c4i * wr,
      );
      let (i1r, i1i, i2r, i2i, i3r, i3i, i4r, i4i) = (
        c1r + c2r,
        c1i + c2i,
        c1r - c2r,
        c1i - c2i,
        c3r + c4r,
        c3i + c4i,
        c3r - c4r,
        c3i - c4i,
      );

      let (i3r, i3i, i4r, i4i) = (
        i3r * w2r - i3i * w2i,
        i3r * w2i + i3i * w2r,
        i4r * w3r - i4i * w3i,
        i4r * w3i + i4i * w3r,
      );

      out_chunk[index] = i1r + i3r;
      out_chunk[index + 1] = i1i + i3i;

      out_chunk[index + block_size] = i2r + i4r;
      out_chunk[index + block_size + 1] = i2i + i4i;

      out_chunk[index + block_jump] = i1r - i3r;
      out_chunk[index + block_jump + 1] = i1i - i3i;

      out_chunk[index + block_jump + block_size] = i2r - i4r;
      out_chunk[index + block_jump + block_size + 1] = i2i - i4i;

      lookup_index += lookup_incr;
      lookup_index_radx4 += lookup_incr_radx4;

      (wr, wi) = (lookup_table[lookup_index], lookup_table[lookup_index + 1]);
      (w2r, w2i) = (
        lookup_table[lookup_index_radx4],
        lookup_table[lookup_index_radx4 + 1],
      );
      (w3r, w3i) = (
        lookup_table[lookup_index_radx4 + mid],
        lookup_table[lookup_index_radx4 + mid + 1],
      );
    }
  });
}

fn radx4ifft_merge_n(
  out: &mut [f32],
  lookup_table: &[f32],
  block_size: usize,
  length_check_lookup: usize,
) {
  let lookup_incr = length_check_lookup << 1;
  let block_jump = block_size << 1;

  out.chunks_mut(block_jump).for_each(|out_chunk| {
    let (mut wr, mut wi) = (1.0, 0.0);
    let mut lookup_index = 0;

    for index in (0..block_size).step_by(2) {
      let (c1r, c1i) = (out_chunk[index], out_chunk[index + 1]);
      let (mut c2r, mut c2i) = (
        out_chunk[index + block_size],
        out_chunk[index + block_size + 1],
      );
      let temp = c2r * wr - c2i * wi;
      c2i = c2r * wi + c2i * wr;
      c2r = temp;

      out_chunk[index] = c1r + c2r;
      out_chunk[index + 1] = c1i + c2i;
      out_chunk[index + block_size] = c1r - c2r;
      out_chunk[index + block_size + 1] = c1i - c2i;

      lookup_index += lookup_incr;
      wr = lookup_table[lookup_index];
      wi = lookup_table[lookup_index + 1];
    }
  });
}

/// Radix-4 fft
///
/// Uses Divide-and-Conquer method, and non-recursive method
#[wasm_bindgen]
pub fn radx4fft(array: &[f32], lookup_table: &[f32]) -> Vec<f32> {
  let index_iter: IndexGen = IndexGen::new(array.len());

  let mut out = Vec::new();
  out.resize_with(array.len() << 1, || 0.0);

  out
    .chunks_mut(2)
    .zip(index_iter.map(|x| array[x]))
    .for_each(|(x, element)| {
      x[0] = element;
    });

  let (mut block_size, length, mut length_check_lookup) =
    (2, array.len(), array.len() >> 1);

  while block_size <= length {
    if (block_size << 1) <= length {
      radx4fft_merge_n_2n(
        &mut out,
        lookup_table,
        block_size,
        length_check_lookup,
      );
    } else {
      radx4fft_merge_n(&mut out, lookup_table, block_size, length_check_lookup);
    }
    block_size <<= 2;
    length_check_lookup >>= 2;
  }

  out
}

/// Radix-4 Inverse-fft
///
/// Uses Divide-and-Conquer method, and non-recursive method
#[wasm_bindgen]
pub fn radx4ifft(array: &[f32], lookup_table: &[f32]) -> Vec<f32> {
  let mut out = Vec::new();
  out.resize_with(array.len(), || 0.0);
  let length = array.len() >> 1;
  let index_iter: IndexGen = IndexGen::new(length);

  index_iter
    .zip(array.chunks(2))
    .for_each(|(rev_index, chunk)| {
      out[rev_index << 1] = chunk[0];
      out[(rev_index << 1) + 1] = chunk[1];
    });

  let (mut block_size, mut length_check_lookup) = (2, length >> 1);

  while block_size <= length {
    if (block_size << 1) <= length {
      radx4ifft_merge_n_2n(
        &mut out,
        lookup_table,
        block_size,
        length_check_lookup,
      );
    } else {
      radx4ifft_merge_n(
        &mut out,
        lookup_table,
        block_size,
        length_check_lookup,
      );
    }
    block_size <<= 2;
    length_check_lookup >>= 2;
  }

  let mut result = Vec::new();
  result.resize_with(length, || 0.0);

  out.chunks(2).zip(result.iter_mut()).for_each(|(c, res)| {
    *res = c[0] / (length as f32);
  });

  result
}

/// Perform Fast Fourier Transform
/// on `n` values of Vec, and returns the floating values
///
/// Uses Divide-and-Conquer method, and non-recursive method
#[wasm_bindgen]
pub fn fft(array: &[f32], lookup_table: &[f32]) -> Vec<f32> {
  let index_iter: IndexGen = IndexGen::new(array.len());

  let mut out = Vec::new();
  out.resize_with(array.len() << 1, || 0.0);

  out
    .chunks_mut(2)
    .zip(index_iter.map(|x| array[x]))
    .for_each(|(x, element)| {
      x[0] = element;
    });

  out.chunks_mut(4).for_each(|out_slice| {
    let c1r = out_slice[0];
    let c1i = out_slice[1];
    let c2r = out_slice[2];
    let c2i = out_slice[3];

    out_slice[0] = c1r + c2r;
    out_slice[1] = c1i + c2i;
    out_slice[2] = c1r - c2r;
    out_slice[3] = c1i - c2i;
  });

  let (mut block_size, length, mut length_check_lookup) =
    (4, array.len(), array.len() >> 2);
  while block_size <= length {
    let lookup_incr = length_check_lookup << 1;
    let block_jump = block_size << 1;

    out.chunks_mut(block_jump).for_each(|out_chunk| {
      let (mut wr, mut wi) = (1.0, 0.0);
      let mut lookup_index = 0;

      for index in (0..block_size).step_by(2) {
        let (c1r, c1i) = (out_chunk[index], out_chunk[index + 1]);
        let (mut c2r, mut c2i) = (
          out_chunk[index + block_size],
          out_chunk[index + block_size + 1],
        );
        let temp = c2r * wr - c2i * wi;
        c2i = c2r * wi + c2i * wr;
        c2r = temp;

        out_chunk[index] = c1r + c2r;
        out_chunk[index + 1] = c1i + c2i;
        out_chunk[index + block_size] = c1r - c2r;
        out_chunk[index + block_size + 1] = c1i - c2i;

        lookup_index += lookup_incr;
        wr = lookup_table[lookup_index];
        wi = -lookup_table[lookup_index + 1];
      }
    });

    block_size <<= 1;
    length_check_lookup >>= 1;
  }

  out
}

/// Perform Fast Fourier Transform
/// on `n` values of Vec, and returns the floating values
///
/// Uses Divide-and-Conquer method, and non-recursive method
#[wasm_bindgen]
pub fn ifft(c_array: &[f32], lookup_table: &[f32]) -> Vec<f32> {
  let mut out = Vec::new();
  out.resize_with(c_array.len(), || 0.0);
  let length = c_array.len() >> 1;
  let index_iter: IndexGen = IndexGen::new(length);

  index_iter
    .zip(c_array.chunks(2))
    .for_each(|(rev_index, chunk)| {
      out[rev_index << 1] = chunk[0];
      out[(rev_index << 1) + 1] = chunk[1];
    });

  out.chunks_mut(4).for_each(|out_slice| {
    let c1r = out_slice[0];
    let c1i = out_slice[1];
    let c2r = out_slice[2];
    let c2i = out_slice[3];

    out_slice[0] = c1r + c2r;
    out_slice[1] = c1i + c2i;
    out_slice[2] = c1r - c2r;
    out_slice[3] = c1i - c2i;
  });

  let (mut block_size, mut length_check_lookup) = (4, length >> 2);
  while block_size <= length {
    let lookup_incr = length_check_lookup << 1;
    let block_jump = block_size << 1;

    out.chunks_mut(block_jump).for_each(|out_chunk| {
      let (mut wr, mut wi) = (1.0, 0.0);
      let mut lookup_index = 0;

      for index in (0..block_size).step_by(2) {
        let (c1r, c1i) = (out_chunk[index], out_chunk[index + 1]);
        let (mut c2r, mut c2i) = (
          out_chunk[index + block_size],
          out_chunk[index + block_size + 1],
        );
        let temp = c2r * wr - c2i * wi;
        c2i = c2r * wi + c2i * wr;
        c2r = temp;

        out_chunk[index] = c1r + c2r;
        out_chunk[index + 1] = c1i + c2i;
        out_chunk[index + block_size] = c1r - c2r;
        out_chunk[index + block_size + 1] = c1i - c2i;

        lookup_index += lookup_incr;
        wr = lookup_table[lookup_index];
        wi = lookup_table[lookup_index + 1];
      }
    });

    block_size <<= 1;
    length_check_lookup >>= 1;
  }

  let mut result = Vec::new();
  result.resize_with(length, || 0.0);

  out.chunks(2).zip(result.iter_mut()).for_each(|(c, res)| {
    *res = c[0] / (length as f32);
  });

  result
}

#[cfg(test)]
mod test {
  use super::*;
  #[test]
  fn test_radx4_merge_2_4() -> Result<(), Box<dyn std::error::Error>> {
    let vec: Vec<f32> = (0..16).map(|c| c as f32).collect();
    let lt = generate_lookup_table(vec.len());

    let t = std::time::Instant::now();
    let fft_vec = fft(&vec, &lt);
    let iff_v = ifft(&fft_vec, &lt);

    let t = std::time::Instant::now();
    let fft_rdx_vec = radx4fft(&vec, &lt);
    let iff = radx4ifft(&fft_rdx_vec, &lt);

    // assert!(iff.iter().zip(iff_v.iter()).all(|(f, s)| ((*f / *s).abs() - 1.0).abs() <= 1e-6));
    println!("{:?}\n\n{:?}", iff_v, iff);

    Ok(())
  }
}
