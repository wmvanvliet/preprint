
write_codes = true;
active_buttons = 2;
begin;

picture {
   default_code = "fixation";
   bitmap {
        filename = "fixation_cross.png";
   };
   x = 0; y = 0;
} fixation;

TEMPLATE "stimulus.tem" {
word file code;
"voo^o" "symbols_voo^o.png" 30;
"opaali" "word_opaali.png" 10;
"dxfxv" "consonants_dxfxv.png" 20;
"odos" "symbols_odos.png" 30;
"siitos" "word_siitos.png" 10;
"d^sd" "symbols_d^sd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_d^sd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"duuri" "word_duuri.png" 10;
"d^v^^" "symbols_d^v^^.png" 30;
"lempo" "word_lempo.png" 10;
"nuha" "word_nuha.png" 10;
"laukka" "word_laukka.png" 10;
"jakaja" "word_jakaja.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ x _ _ _ _" "word_jakaja_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"räme" "word_räme.png" 10;
"kyhmy" "word_kyhmy.png" 10;
"ovo^s" "symbols_ovo^s.png" 30;
"vsd^ds" "symbols_vsd^ds.png" 30;
"huorin" "word_huorin.png" 10;
"pidot" "word_pidot.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _" "word_pidot_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jxfm" "consonants_jxfm.png" 20;
"vwrptk" "consonants_vwrptk.png" 20;
"uuhi" "word_uuhi.png" 10;
"akti" "word_akti.png" 10;
"sdvdvd" "symbols_sdvdvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "symbols_sdvdvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tämä" "word_tämä.png" 10;
"rmmrh" "consonants_rmmrh.png" 20;
"kussa" "word_kussa.png" 10;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fbtsr" "consonants_fbtsr.png" 20;
"vtkdtx" "consonants_vtkdtx.png" 20;
"rjfcdx" "consonants_rjfcdx.png" 20;
"tyköä" "word_tyköä.png" 10;
"lxtgwm" "consonants_lxtgwm.png" 20;
"jbntmc" "consonants_jbntmc.png" 20;
"xkfvmb" "consonants_xkfvmb.png" 20;
"sovo" "symbols_sovo.png" 30;
"ltqrr" "consonants_ltqrr.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ j _" "consonants_ltqrr_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tkcqd" "consonants_tkcqd.png" 20;
"gljt" "consonants_gljt.png" 20;
"silaus" "word_silaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "word_silaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ovvds" "symbols_ovvds.png" 30;
"lähi" "word_lähi.png" 10;
"qgjwzq" "consonants_qgjwzq.png" 20;
"pitäen" "word_pitäen.png" 10;
"lotja" "word_lotja.png" 10;
"^so^d" "symbols_^so^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_^so^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kiuas" "word_kiuas.png" 10;
"tczhj" "consonants_tczhj.png" 20;
"säie" "word_säie.png" 10;
"ajos" "word_ajos.png" 10;
"grgdd" "consonants_grgdd.png" 20;
"jymy" "word_jymy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_jymy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"almu" "word_almu.png" 10;
"mwgz" "consonants_mwgz.png" 20;
"tcftcg" "consonants_tcftcg.png" 20;
"rfclk" "consonants_rfclk.png" 20;
"xgtt" "consonants_xgtt.png" 20;
"oinas" "word_oinas.png" 10;
"pokeri" "word_pokeri.png" 10;
"wnnz" "consonants_wnnz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ n _ _" "consonants_wnnz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xwnh" "consonants_xwnh.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"whpw" "consonants_whpw.png" 20;
"ässä" "word_ässä.png" 10;
"näppy" "word_näppy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _" "word_näppy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fotoni" "word_fotoni.png" 10;
"d^dvdd" "symbols_d^dvdd.png" 30;
"bhsx" "consonants_bhsx.png" 20;
"sysi" "word_sysi.png" 10;
"svvssv" "symbols_svvssv.png" 30;
"uoma" "word_uoma.png" 10;
"psalmi" "word_psalmi.png" 10;
"fwgntw" "consonants_fwgntw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ w _ _ _ _" "consonants_fwgntw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dwzsrc" "consonants_dwzsrc.png" 20;
"menijä" "word_menijä.png" 10;
"rukki" "word_rukki.png" 10;
"s^od" "symbols_s^od.png" 30;
"nirso" "word_nirso.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _ _" "word_nirso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdosd" "symbols_sdosd.png" 30;
"osvs^d" "symbols_osvs^d.png" 30;
"tyrä" "word_tyrä.png" 10;
"dqkcl" "consonants_dqkcl.png" 20;
"^s^s" "symbols_^s^s.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _" "symbols_^s^s_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dvsv" "symbols_dvsv.png" 30;
"qpvbs" "consonants_qpvbs.png" 20;
"rwfcq" "consonants_rwfcq.png" 20;
"iäti" "word_iäti.png" 10;
"vipu" "word_vipu.png" 10;
"odote" "word_odote.png" 10;
"dsdoo^" "symbols_dsdoo^.png" 30;
"tcfwdr" "consonants_tcfwdr.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ c _ _ _ _" "consonants_tcfwdr_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"czkdbs" "consonants_czkdbs.png" 20;
"vs^vds" "symbols_vs^vds.png" 30;
"hioa" "word_hioa.png" 10;
"säle" "word_säle.png" 10;
"gthb" "consonants_gthb.png" 20;
"^sodvv" "symbols_^sodvv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "symbols_^sodvv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vs^oo" "symbols_vs^oo.png" 30;
"kopio" "word_kopio.png" 10;
"mxnhh" "consonants_mxnhh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ z" "consonants_mxnhh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kerubi" "word_kerubi.png" 10;
"katodi" "word_katodi.png" 10;
"^vvs^s" "symbols_^vvs^s.png" 30;
"mäntä" "word_mäntä.png" 10;
"krvvpr" "consonants_krvvpr.png" 20;
"jdfjs" "consonants_jdfjs.png" 20;
"wcjrjq" "consonants_wcjrjq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ c _ _ _" "consonants_wcjrjq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"afasia" "word_afasia.png" 10;
"druidi" "word_druidi.png" 10;
"nurja" "word_nurja.png" 10;
"viipyä" "word_viipyä.png" 10;
"ositus" "word_ositus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"ö _ _ _ _ _" "word_ositus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"bzhkd" "consonants_bzhkd.png" 20;
"känsä" "word_känsä.png" 10;
"kyteä" "word_kyteä.png" 10;
"fuksi" "word_fuksi.png" 10;
"oo^vdv" "symbols_oo^vdv.png" 30;
"ksrwvk" "consonants_ksrwvk.png" 20;
"dovv^" "symbols_dovv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ o _ _ _" "symbols_dovv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mgttwx" "consonants_mgttwx.png" 20;
"eliö" "word_eliö.png" 10;
"puida" "word_puida.png" 10;
"gongi" "word_gongi.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"ylkä" "word_ylkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "word_ylkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"itää" "word_itää.png" 10;
"uuni" "word_uuni.png" 10;
"riimu" "word_riimu.png" 10;
"^oddo" "symbols_^oddo.png" 30;
"luusto" "word_luusto.png" 10;
"tuohus" "word_tuohus.png" 10;
"odvs^" "symbols_odvs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _ _" "symbols_odvs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dssddd" "symbols_dssddd.png" 30;
"bwdz" "consonants_bwdz.png" 20;
"rrggj" "consonants_rrggj.png" 20;
"gxqtx" "consonants_gxqtx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ x" "consonants_gxqtx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"diodi" "word_diodi.png" 10;
"ilmi" "word_ilmi.png" 10;
"d^^do" "symbols_d^^do.png" 30;
"xpfpj" "consonants_xpfpj.png" 20;
"nieriä" "word_nieriä.png" 10;
"sikhi" "word_sikhi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"h _ _ _ _" "word_sikhi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hqzll" "consonants_hqzll.png" 20;
"kolvi" "word_kolvi.png" 10;
"kphwjz" "consonants_kphwjz.png" 20;
"vovso" "symbols_vovso.png" 30;
"kihu" "word_kihu.png" 10;
"emätin" "word_emätin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"c _ _ _ _ _" "word_emätin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"odo^" "symbols_odo^.png" 30;
"pmxwht" "consonants_pmxwht.png" 20;
"vaje" "word_vaje.png" 10;
"qwjvhz" "consonants_qwjvhz.png" 20;
"sdvsv" "symbols_sdvsv.png" 30;
"^^svd" "symbols_^^svd.png" 30;
"solmio" "word_solmio.png" 10;
"suippo" "word_suippo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ p _ _" "word_suippo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"doosv" "symbols_doosv.png" 30;
"^ss^" "symbols_^ss^.png" 30;
"piip" "word_piip.png" 10;
"motata" "word_motata.png" 10;
"voov^v" "symbols_voov^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ^ _" "symbols_voov^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dfdzj" "consonants_dfdzj.png" 20;
"^^^os" "symbols_^^^os.png" 30;
"jqdsmj" "consonants_jqdsmj.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"d^oood" "symbols_d^oood.png" 30;
"tvzdg" "consonants_tvzdg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "consonants_tvzdg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ähky" "word_ähky.png" 10;
"lakea" "word_lakea.png" 10;
"ovdvd" "symbols_ovdvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "symbols_ovdvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kolhia" "word_kolhia.png" 10;
"lgtjb" "consonants_lgtjb.png" 20;
"yöpyä" "word_yöpyä.png" 10;
"jäte" "word_jäte.png" 10;
"dvqdj" "consonants_dvqdj.png" 20;
"kuje" "word_kuje.png" 10;
"oosvsd" "symbols_oosvsd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _ _" "symbols_oosvsd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häpy" "word_häpy.png" 10;
"^ddv" "symbols_^ddv.png" 30;
"räntä" "word_räntä.png" 10;
"klaava" "word_klaava.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"ampuja" "word_ampuja.png" 10;
"s^^d" "symbols_s^^d.png" 30;
"kaapia" "word_kaapia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"k _ _ _ _ _" "word_kaapia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"otsoni" "word_otsoni.png" 10;
"s^vds^" "symbols_s^vds^.png" 30;
"^dso" "symbols_^dso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _" "symbols_^dso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ahjo" "word_ahjo.png" 10;
"vssd" "symbols_vssd.png" 30;
"miten" "word_miten.png" 10;
"bxhl" "consonants_bxhl.png" 20;
"faksi" "word_faksi.png" 10;
"zfjxqk" "consonants_zfjxqk.png" 20;
"erkani" "word_erkani.png" 10;
"lohi" "word_lohi.png" 10;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"möly" "word_möly.png" 10;
"ssss" "symbols_ssss.png" 30;
"uute" "word_uute.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _" "word_uute_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"salvaa" "word_salvaa.png" 10;
"vdvv" "symbols_vdvv.png" 30;
"ztlrlf" "consonants_ztlrlf.png" 20;
"zsgj" "consonants_zsgj.png" 20;
"salaus" "word_salaus.png" 10;
"tenä" "word_tenä.png" 10;
"hmff" "consonants_hmff.png" 20;
"poru" "word_poru.png" 10;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dsovsd" "symbols_dsovsd.png" 30;
"loraus" "word_loraus.png" 10;
"syaani" "word_syaani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "word_syaani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nurin" "word_nurin.png" 10;
"isyys" "word_isyys.png" 10;
"harjus" "word_harjus.png" 10;
"osinko" "word_osinko.png" 10;
"kysta" "word_kysta.png" 10;
"pesula" "word_pesula.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_pesula_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"älytä" "word_älytä.png" 10;
"s^s^s" "symbols_s^s^s.png" 30;
"silsa" "word_silsa.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"vdod^" "symbols_vdod^.png" 30;
"bjwqk" "consonants_bjwqk.png" 20;
"blkxwz" "consonants_blkxwz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ w _" "consonants_blkxwz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tykö" "word_tykö.png" 10;
"äimä" "word_äimä.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"osdod^" "symbols_osdod^.png" 30;
"s^o^^v" "symbols_s^o^^v.png" 30;
"urut" "word_urut.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_urut_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^v^" "symbols_v^v^.png" 30;
"näkö" "word_näkö.png" 10;
"dovdd" "symbols_dovdd.png" 30;
"dvsvos" "symbols_dvsvos.png" 30;
"ripsi" "word_ripsi.png" 10;
"myyty" "word_myyty.png" 10;
"tyvi" "word_tyvi.png" 10;
"s^dv^" "symbols_s^dv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_s^dv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dfszzp" "consonants_dfszzp.png" 20;
"dlgt" "consonants_dlgt.png" 20;
"rulla" "word_rulla.png" 10;
"uima" "word_uima.png" 10;
"s^vo" "symbols_s^vo.png" 30;
"äänes" "word_äänes.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ä _ _ _" "word_äänes_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"noppa" "word_noppa.png" 10;
"mkfz" "consonants_mkfz.png" 20;
"korren" "word_korren.png" 10;
"estyä" "word_estyä.png" 10;
"^osd" "symbols_^osd.png" 30;
"uivelo" "word_uivelo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ o" "word_uivelo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rrwj" "consonants_rrwj.png" 20;
"svvd" "symbols_svvd.png" 30;
"orpo" "word_orpo.png" 10;
"hamaan" "word_hamaan.png" 10;
"juhta" "word_juhta.png" 10;
"xxqhnz" "consonants_xxqhnz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ q" "consonants_xxqhnz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"spmb" "consonants_spmb.png" 20;
"^svs^" "symbols_^svs^.png" 30;
"köli" "word_köli.png" 10;
"v^vsov" "symbols_v^vsov.png" 30;
"terska" "word_terska.png" 10;
"s^od^" "symbols_s^od^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _ _" "symbols_s^od^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hxjlgq" "consonants_hxjlgq.png" 20;
"do^d" "symbols_do^d.png" 30;
"vuoka" "word_vuoka.png" 10;
"kortti" "word_kortti.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ c" "consonants_zpcqc_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"päkiä" "word_päkiä.png" 10;
"kytkin" "word_kytkin.png" 10;
"qckq" "consonants_qckq.png" 20;
"zztr" "consonants_zztr.png" 20;
"^d^vs" "symbols_^d^vs.png" 30;
"tlzr" "consonants_tlzr.png" 20;
"sävy" "word_sävy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ e _" "word_sävy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gchqcz" "consonants_gchqcz.png" 20;
"zdvv" "consonants_zdvv.png" 20;
"voss" "symbols_voss.png" 30;
"hihna" "word_hihna.png" 10;
"rämä" "word_rämä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ h _ _" "word_rämä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kopina" "word_kopina.png" 10;
"vdso" "symbols_vdso.png" 30;
"sv^vvs" "symbols_sv^vvs.png" 30;
"^vod" "symbols_^vod.png" 30;
"torium" "word_torium.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ m" "word_torium_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hjqc" "consonants_hjqc.png" 20;
"gsdx" "consonants_gsdx.png" 20;
"txhs" "consonants_txhs.png" 20;
"pyörö" "word_pyörö.png" 10;
"uiva" "word_uiva.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_uiva_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sarake" "word_sarake.png" 10;
"^vovvs" "symbols_^vovvs.png" 30;
"hajan" "word_hajan.png" 10;
"kohu" "word_kohu.png" 10;
"lemu" "word_lemu.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v _" "consonants_ndsvw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"arpoa" "word_arpoa.png" 10;
"pöty" "word_pöty.png" 10;
"häkä" "word_häkä.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"uuma" "word_uuma.png" 10;
"ssods" "symbols_ssods.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_ssods_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tdgw" "consonants_tdgw.png" 20;
"särkyä" "word_särkyä.png" 10;
"häät" "word_häät.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"jaos" "word_jaos.png" 10;
"ääriin" "word_ääriin.png" 10;
"vsd^" "symbols_vsd^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_vsd^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"osuma" "word_osuma.png" 10;
"btmk" "consonants_btmk.png" 20;
"itiö" "word_itiö.png" 10;
"xggtw" "consonants_xggtw.png" 20;
"vzkt" "consonants_vzkt.png" 20;
"lcrzjb" "consonants_lcrzjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ b _ _" "consonants_lcrzjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säiky" "word_säiky.png" 10;
"^d^^d" "symbols_^d^^d.png" 30;
"ratamo" "word_ratamo.png" 10;
"hormi" "word_hormi.png" 10;
"sopu" "word_sopu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ c _ _" "word_sopu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ripeä" "word_ripeä.png" 10;
"vv^^" "symbols_vv^^.png" 30;
"sodvdv" "symbols_sodvdv.png" 30;
"jaardi" "word_jaardi.png" 10;
"vahaus" "word_vahaus.png" 10;
"kuskus" "word_kuskus.png" 10;
"pesin" "word_pesin.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ o" "symbols_^vd^vd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ds^^o" "symbols_ds^^o.png" 30;
"o^ss" "symbols_o^ss.png" 30;
"rapsi" "word_rapsi.png" 10;
"gnbtn" "consonants_gnbtn.png" 20;
"hautua" "word_hautua.png" 10;
"os^ood" "symbols_os^ood.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _ _ _" "symbols_os^ood_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvo^" "symbols_vvo^.png" 30;
"so^^v" "symbols_so^^v.png" 30;
"stoola" "word_stoola.png" 10;
"^^od" "symbols_^^od.png" 30;
"seimi" "word_seimi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m _" "word_seimi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvvd" "symbols_vvvd.png" 30;
"karies" "word_karies.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"äänne" "word_äänne.png" 10;
"suti" "word_suti.png" 10;
"vo^^" "symbols_vo^^.png" 30;
"qvkjt" "consonants_qvkjt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"q _ _ _ _" "consonants_qvkjt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^d^oos" "symbols_^d^oos.png" 30;
"shiia" "word_shiia.png" 10;
"zkwnr" "consonants_zkwnr.png" 20;
"kaste" "word_kaste.png" 10;
"säkä" "word_säkä.png" 10;
"jänne" "word_jänne.png" 10;
"nisä" "word_nisä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_nisä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"someen" "word_someen.png" 10;
"täällä" "word_täällä.png" 10;
"vovdsv" "symbols_vovdsv.png" 30;
"nide" "word_nide.png" 10;
"dv^od" "symbols_dv^od.png" 30;
"lhqt" "consonants_lhqt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t" "consonants_lhqt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
};
