-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Waktu pembuatan: 21 Apr 2023 pada 16.56
-- Versi server: 10.4.27-MariaDB
-- Versi PHP: 7.4.33

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `absensi_wajah`
--

DELIMITER $$
--
-- Prosedur
--
CREATE DEFINER=`root`@`localhost` PROCEDURE `sp_add_pegawai` (IN `p_nama` VARCHAR(255), IN `p_file_gambar` VARCHAR(255))   BEGIN
    INSERT INTO pegawai(nama, file_gambar)
    VALUES (p_nama, p_file_gambar);
END$$

CREATE DEFINER=`root`@`localhost` PROCEDURE `sp_delete_pegawai` (IN `p_id` INT)   BEGIN
    DELETE FROM pegawai
    WHERE id = p_id;
END$$

CREATE DEFINER=`root`@`localhost` PROCEDURE `sp_edit_pegawai` (IN `p_id` INT, IN `p_nama` VARCHAR(255), IN `p_file_gambar` VARCHAR(255))   BEGIN
    UPDATE pegawai
    SET nama = p_nama, file_gambar = p_file_gambar
    WHERE id = p_id;
END$$

DELIMITER ;

-- --------------------------------------------------------

--
-- Struktur dari tabel `absen`
--

CREATE TABLE `absen` (
  `no` int(11) NOT NULL,
  `id_pegawai` int(11) NOT NULL,
  `waktu_absen` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `tipe_absen` enum('pagi','siang','sore') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `admin`
--

INSERT INTO `admin` (`id`, `username`, `password`) VALUES
(1, 'admin', '123');

-- --------------------------------------------------------

--
-- Struktur dari tabel `pegawai`
--

CREATE TABLE `pegawai` (
  `id` int(11) NOT NULL,
  `nama` varchar(255) NOT NULL,
  `file_gambar` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `pegawai`
--

INSERT INTO `pegawai` (`id`, `nama`, `file_gambar`) VALUES
(0, 'aa', '0.jpg'),
(1, 'bb', '1.jpg'),
(2, 'cc', '2.jpg'),
(3, 'dd', '3.jpg'),
(4, 'asnf', '4.jpg'),
(5, 'aisn', '5.jpg'),
(6, 'safnc', '6.jpg'),
(7, 'asjc', '7.jpg'),
(8, 'asjhc', '8.jpg'),
(9, 'jascc', '9.jpg'),
(10, 'ascx', '10.jpg'),
(11, 'sasco', '11.jpg');

-- --------------------------------------------------------

--
-- Stand-in struktur untuk tampilan `tb_absen`
-- (Lihat di bawah untuk tampilan aktual)
--
CREATE TABLE `tb_absen` (
`id_pegawai` int(11)
,`nama` varchar(255)
,`tanggal` date
,`absen_masuk` time
,`absen_keluar` time
);

-- --------------------------------------------------------

--
-- Struktur untuk view `tb_absen`
--
DROP TABLE IF EXISTS `tb_absen`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `tb_absen`  AS SELECT `a`.`id_pegawai` AS `id_pegawai`, `p`.`nama` AS `nama`, cast(`a`.`waktu_absen` as date) AS `tanggal`, cast(min(`a`.`waktu_absen`) as time) AS `absen_masuk`, cast(max(`a`.`waktu_absen`) as time) AS `absen_keluar` FROM (`absen` `a` join `pegawai` `p` on(`a`.`id_pegawai` = `p`.`id`)) GROUP BY `a`.`id_pegawai`, cast(`a`.`waktu_absen` as date)  ;

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `absen`
--
ALTER TABLE `absen`
  ADD PRIMARY KEY (`no`),
  ADD KEY `id_pegawai` (`id_pegawai`);

--
-- Indeks untuk tabel `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `pegawai`
--
ALTER TABLE `pegawai`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `absen`
--
ALTER TABLE `absen`
  MODIFY `no` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=30;

--
-- AUTO_INCREMENT untuk tabel `admin`
--
ALTER TABLE `admin`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT untuk tabel `pegawai`
--
ALTER TABLE `pegawai`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=71190488;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `absen`
--
ALTER TABLE `absen`
  ADD CONSTRAINT `absen_ibfk_1` FOREIGN KEY (`id_pegawai`) REFERENCES `pegawai` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
