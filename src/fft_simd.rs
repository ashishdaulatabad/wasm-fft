use wasm_bindgen_test::*;

use crate::index_generator::IndexGen;

#[derive(Debug, Clone)]
pub struct CArray {
  pub r: Vec<f32>,
  pub i: Vec<f32>,
}

impl CArray {
  #[inline]
  pub fn new(len: usize) -> Self {
    let mut r = Vec::new();
    r.resize_with(len, || 0.0);

    let mut i = Vec::new();
    i.resize_with(len, || 0.0);

    Self { r, i }
  }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn merge_2(output: &mut CArray) {
  output.r.chunks_mut(8).zip(output.i.chunks_mut(8)).for_each(
    |(out_r, out_i)| {
      let c1r =
        core::arch::wasm32::f32x4(out_r[0], out_r[2], out_r[4], out_r[6]);
      let c2r =
        core::arch::wasm32::f32x4(out_r[1], out_r[3], out_r[5], out_r[7]);
      let c1i =
        core::arch::wasm32::f32x4(out_i[0], out_i[2], out_i[4], out_i[6]);
      let c2i =
        core::arch::wasm32::f32x4(out_i[1], out_i[3], out_i[5], out_i[7]);

      let sum_r = core::arch::wasm32::f32x4_add(c1r, c2r);
      let sum_i = core::arch::wasm32::f32x4_add(c1i, c2i);
      let diff_r = core::arch::wasm32::f32x4_sub(c1r, c2r);
      let diff_i = core::arch::wasm32::f32x4_sub(c1i, c2i);

      (out_r[0], out_r[2], out_r[4], out_r[6]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_r),
      );

      (out_r[1], out_r[3], out_r[5], out_r[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_r),
      );

      (out_i[0], out_i[2], out_i[4], out_i[6]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_i),
      );

      (out_i[1], out_i[3], out_i[5], out_i[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_i),
      );
    },
  );
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn merge_4(output: &mut CArray) {
  output.r.chunks_mut(8).zip(output.i.chunks_mut(8)).for_each(
    |(out_r, out_i)| {
      let c1r =
        core::arch::wasm32::f32x4(out_r[0], out_r[1], out_r[4], out_r[5]);
      let c2r =
        core::arch::wasm32::f32x4(out_r[2], out_i[3], out_r[6], out_i[7]);
      let c1i =
        core::arch::wasm32::f32x4(out_i[0], out_i[1], out_i[4], out_i[5]);
      let c2i =
        core::arch::wasm32::f32x4(out_i[2], -out_r[3], out_i[6], -out_r[7]);

      let sum_r = core::arch::wasm32::f32x4_add(c1r, c2r);
      let sum_i = core::arch::wasm32::f32x4_add(c1i, c2i);
      let diff_r = core::arch::wasm32::f32x4_sub(c1r, c2r);
      let diff_i = core::arch::wasm32::f32x4_sub(c1i, c2i);

      (out_r[0], out_r[1], out_r[4], out_r[5]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_r),
      );

      (out_r[2], out_r[3], out_r[6], out_r[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_r),
      );

      (out_i[0], out_i[1], out_i[4], out_i[5]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_i),
      );

      (out_i[2], out_i[3], out_i[6], out_i[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_i),
      );
    },
  );
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn merge_inverse_4(output: &mut CArray) {
  output.r.chunks_mut(8).zip(output.i.chunks_mut(8)).for_each(
    |(out_r, out_i)| {
      let c1r =
        core::arch::wasm32::f32x4(out_r[0], out_r[1], out_r[4], out_r[5]);
      let c2r =
        core::arch::wasm32::f32x4(out_r[2], -out_i[3], out_r[6], -out_i[7]);
      let c1i =
        core::arch::wasm32::f32x4(out_i[0], out_i[1], out_i[4], out_i[5]);
      let c2i =
        core::arch::wasm32::f32x4(out_i[2], out_r[3], out_i[6], out_r[7]);

      let sum_r = core::arch::wasm32::f32x4_add(c1r, c2r);
      let sum_i = core::arch::wasm32::f32x4_add(c1i, c2i);
      let diff_r = core::arch::wasm32::f32x4_sub(c1r, c2r);
      let diff_i = core::arch::wasm32::f32x4_sub(c1i, c2i);

      (out_r[0], out_r[1], out_r[4], out_r[5]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_r),
      );

      (out_r[2], out_r[3], out_r[6], out_r[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_r),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_r),
      );

      (out_i[0], out_i[1], out_i[4], out_i[5]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(sum_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(sum_i),
      );

      (out_i[2], out_i[3], out_i[6], out_i[7]) = (
        core::arch::wasm32::f32x4_extract_lane::<0>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<1>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<2>(diff_i),
        core::arch::wasm32::f32x4_extract_lane::<3>(diff_i),
      );
    },
  );
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn merge_n(
  output: &mut CArray,
  lookup_table: &CArray,
  block_size: usize,
  length_check_lookup: usize,
) {
  let block_size_half = block_size >> 1;
  let lookup_incr = length_check_lookup;

  output
    .r
    .chunks_mut(block_size)
    .zip(output.i.chunks_mut(block_size))
    .for_each(|(out_r, out_i)| {
      let mut lookup_index = 0;

      let (mut wr, mut wi) = (
        core::arch::wasm32::f32x4(
          lookup_table.r[lookup_index],
          lookup_table.r[lookup_index + lookup_incr],
          lookup_table.r[lookup_index + 2 * lookup_incr],
          lookup_table.r[lookup_index + 3 * lookup_incr],
        ),
        core::arch::wasm32::f32x4(
          -lookup_table.i[lookup_index],
          -lookup_table.i[lookup_index + lookup_incr],
          -lookup_table.i[lookup_index + 2 * lookup_incr],
          -lookup_table.i[lookup_index + 3 * lookup_incr],
        ),
      );

      for i in (0..block_size_half).step_by(4) {
        let c1r = core::arch::wasm32::v128_load(
          out_r.as_ptr().add(i) as *const core::arch::wasm32::v128
        );
        let mut c2r = core::arch::wasm32::v128_load(
          out_r.as_ptr().add(i + block_size_half)
            as *const core::arch::wasm32::v128,
        );
        let c1i = core::arch::wasm32::v128_load(
          out_i.as_ptr().add(i) as *const core::arch::wasm32::v128
        );
        let mut c2i = core::arch::wasm32::v128_load(
          out_i.as_ptr().add(i + block_size_half)
            as *const core::arch::wasm32::v128,
        );

        (c2r, c2i) = (
          core::arch::wasm32::f32x4_sub(
            core::arch::wasm32::f32x4_mul(c2r, wr),
            core::arch::wasm32::f32x4_mul(c2i, wi),
          ),
          core::arch::wasm32::f32x4_add(
            core::arch::wasm32::f32x4_mul(c2r, wi),
            core::arch::wasm32::f32x4_mul(c2i, wr),
          ),
        );

        let sum_r = core::arch::wasm32::f32x4_add(c1r, c2r);
        let sum_i = core::arch::wasm32::f32x4_add(c1i, c2i);
        let diff_r = core::arch::wasm32::f32x4_sub(c1r, c2r);
        let diff_i = core::arch::wasm32::f32x4_sub(c1i, c2i);

        core::arch::wasm32::v128_store(
          out_r.as_mut_ptr().add(i) as *mut core::arch::wasm32::v128,
          sum_r,
        );

        core::arch::wasm32::v128_store(
          out_r.as_mut_ptr().add(i + block_size_half)
            as *mut core::arch::wasm32::v128,
          diff_r,
        );

        core::arch::wasm32::v128_store(
          out_i.as_mut_ptr().add(i) as *mut core::arch::wasm32::v128,
          sum_i,
        );

        core::arch::wasm32::v128_store(
          out_i.as_mut_ptr().add(i + block_size_half)
            as *mut core::arch::wasm32::v128,
          diff_i,
        );

        lookup_index += lookup_incr * 4;
        wr = core::arch::wasm32::f32x4(
          lookup_table.r[lookup_index],
          lookup_table.r[lookup_index + lookup_incr],
          lookup_table.r[lookup_index + 2 * lookup_incr],
          lookup_table.r[lookup_index + 3 * lookup_incr],
        );
        wi = core::arch::wasm32::f32x4(
          -lookup_table.i[lookup_index],
          -lookup_table.i[lookup_index + lookup_incr],
          -lookup_table.i[lookup_index + 2 * lookup_incr],
          -lookup_table.i[lookup_index + 3 * lookup_incr],
        );
      }
    });
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn merge_inverse_n(
  output: &mut CArray,
  lookup_table: &CArray,
  block_size: usize,
  length_check_lookup: usize,
) {
  let block_size_half = block_size >> 1;
  let lookup_incr = length_check_lookup;

  output
    .r
    .chunks_mut(block_size)
    .zip(output.i.chunks_mut(block_size))
    .for_each(|(out_r, out_i)| {
      let mut lookup_index = 0;

      let (mut wr, mut wi) = (
        core::arch::wasm32::f32x4(
          lookup_table.r[lookup_index],
          lookup_table.r[lookup_index + lookup_incr],
          lookup_table.r[lookup_index + 2 * lookup_incr],
          lookup_table.r[lookup_index + 3 * lookup_incr],
        ),
        core::arch::wasm32::f32x4(
          lookup_table.i[lookup_index],
          lookup_table.i[lookup_index + lookup_incr],
          lookup_table.i[lookup_index + 2 * lookup_incr],
          lookup_table.i[lookup_index + 3 * lookup_incr],
        ),
      );

      for i in (0..block_size_half).step_by(4) {
        let c1r = core::arch::wasm32::v128_load(
          out_r.as_ptr().add(i) as *const core::arch::wasm32::v128
        );
        let mut c2r = core::arch::wasm32::v128_load(
          out_r.as_ptr().add(i + block_size_half)
            as *const core::arch::wasm32::v128,
        );
        let c1i = core::arch::wasm32::v128_load(
          out_i.as_ptr().add(i) as *const core::arch::wasm32::v128
        );
        let mut c2i = core::arch::wasm32::v128_load(
          out_i.as_ptr().add(i + block_size_half)
            as *const core::arch::wasm32::v128,
        );

        (c2r, c2i) = (
          core::arch::wasm32::f32x4_sub(
            core::arch::wasm32::f32x4_mul(c2r, wr),
            core::arch::wasm32::f32x4_mul(c2i, wi),
          ),
          core::arch::wasm32::f32x4_add(
            core::arch::wasm32::f32x4_mul(c2r, wi),
            core::arch::wasm32::f32x4_mul(c2i, wr),
          ),
        );

        let sum_r = core::arch::wasm32::f32x4_add(c1r, c2r);
        let sum_i = core::arch::wasm32::f32x4_add(c1i, c2i);
        let diff_r = core::arch::wasm32::f32x4_sub(c1r, c2r);
        let diff_i = core::arch::wasm32::f32x4_sub(c1i, c2i);

        core::arch::wasm32::v128_store(
          out_r.as_mut_ptr().add(i) as *mut core::arch::wasm32::v128,
          sum_r,
        );

        core::arch::wasm32::v128_store(
          out_r.as_mut_ptr().add(i + block_size_half)
            as *mut core::arch::wasm32::v128,
          diff_r,
        );

        core::arch::wasm32::v128_store(
          out_i.as_mut_ptr().add(i) as *mut core::arch::wasm32::v128,
          sum_i,
        );

        core::arch::wasm32::v128_store(
          out_i.as_mut_ptr().add(i + block_size_half)
            as *mut core::arch::wasm32::v128,
          diff_i,
        );

        lookup_index += lookup_incr * 4;

        wr = core::arch::wasm32::f32x4(
          lookup_table.r[lookup_index],
          lookup_table.r[lookup_index + lookup_incr],
          lookup_table.r[lookup_index + 2 * lookup_incr],
          lookup_table.r[lookup_index + 3 * lookup_incr],
        );
        wi = core::arch::wasm32::f32x4(
          lookup_table.i[lookup_index],
          lookup_table.i[lookup_index + lookup_incr],
          lookup_table.i[lookup_index + 2 * lookup_incr],
          lookup_table.i[lookup_index + 3 * lookup_incr],
        );
      }
    });
}

pub fn fft_simd_inplace(
  input: &[f32],
  lookup_table: &CArray,
  output: &mut CArray,
) {
  let len = input.len();
  let index_iter: IndexGen = IndexGen::new(len);
  output.i.fill(0.0);

  output.r.iter_mut().zip(index_iter).for_each(|(r, index)| {
    *r = input[index];
  });
  unsafe {
    merge_2(output);
    // Placeholder for the actual FFT implementation
    // (This is where you would implement the FFT algorithm)
    merge_4(output);

    let (mut block_size, length, mut length_check_lookup) =
      (8, len, output.r.len() >> 3);

    while block_size <= length {
      merge_n(output, lookup_table, block_size, length_check_lookup);
      block_size <<= 1;
      length_check_lookup >>= 1;
    }
  }
}

pub fn fft_simd(input: &[f32], lookup_table: &CArray) -> CArray {
  let len = input.len();
  let mut output = CArray::new(len);

  fft_simd_inplace(input, lookup_table, &mut output);
  output
}

pub fn ifft_simd_inplace(
  input: &CArray,
  lookup_table: &CArray,
  output: &mut CArray,
) {
  let len = input.r.len();
  let index_iter: IndexGen = IndexGen::new(len);

  output.r.iter_mut().zip(index_iter).enumerate().for_each(
    |(i, (r, index))| {
      *r = input.r[index];
      output.i[index] = input.i[i];
    },
  );

  unsafe {
    // Works the same as fft_simd
    merge_2(output);
    merge_inverse_4(output);

    let (mut block_size, length, mut length_check_lookup) =
      (8, len, output.r.len() >> 3);

    // Placeholder for the actual FFT implementation
    // (This is where you would implement the FFT algorithm)
    while block_size <= length {
      merge_inverse_n(output, lookup_table, block_size, length_check_lookup);
      block_size <<= 1;
      length_check_lookup >>= 1;
    }
  }

  // FFT imaginary part should be negated
  output.r.iter_mut().for_each(|r| *r /= len as f32);
}

pub fn ifft_simd(input: &CArray, lookup_table: &CArray) -> Vec<f32> {
  let len = input.r.len();
  let mut output = CArray::new(len);

  ifft_simd_inplace(input, lookup_table, &mut output);
  output.r
}

#[wasm_bindgen_test]
fn test_fft_simd() {
  #[cfg(target_arch = "wasm32")]
  use crate::{generate_lookup_table, radx4fft};

  let size = 8192;
  let s_f32 = size as f32;

  let mut lookup_table = CArray::new(size);

  lookup_table.r.iter_mut().enumerate().for_each(|(i, val)| {
    *val = (2.0 * std::f32::consts::PI * i as f32 / s_f32).cos();
  });
  lookup_table.i.iter_mut().enumerate().for_each(|(i, val)| {
    *val = (2.0 * std::f32::consts::PI * i as f32 / s_f32).sin();
  });

  let other_lookup_table = generate_lookup_table(size);
  let input: Vec<f32> = (0..size).map(|x| x as f32).collect();

  // Implement radx4fft simd.
  let t = web_time::Instant::now();
  let mut c = 0;
  while t.elapsed().as_secs_f32() < 1.0 {
    let result = fft_simd(&input, &lookup_table);
    c += 1;
  }
  console_log!("FFT SIMD iterations in 1 second: {}", c);

  let result = fft_simd(&input, &lookup_table);

  let t = web_time::Instant::now();
  c = 0;

  while t.elapsed().as_secs_f32() < 1.0 {
    let _inv = radx4fft(&input, &other_lookup_table);
    c += 1;
  }
  console_log!("FFT Other iterations in 1 second: {}", c);
  // let other_fft = radx4fft(&input, &other_lookup_table);
  // let _other_ifft = radx4ifft(&other_fft, &other_lookup_table);
  let elapsed = t.elapsed();
  assert!(1 == 2);

  // _inv.iter().zip(_other_ifft.iter()).for_each(|(a, b)| {
  //   let diff = (a - b).abs();
  //   assert!(diff < 1e-5);
  // });
}
