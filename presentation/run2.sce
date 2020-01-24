
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
"^^svd" "symbols_^^svd.png" 30;
"wcjrjq" "consonants_wcjrjq.png" 20;
"vovvvv" "symbols_vovvvv.png" 30;
"mgttwx" "consonants_mgttwx.png" 20;
"tcftcg" "consonants_tcftcg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ c _ _ _ _" "consonants_tcftcg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"osuma" "word_osuma.png" 10;
"ylkä" "word_ylkä.png" 10;
"viipyä" "word_viipyä.png" 10;
"rämä" "word_rämä.png" 10;
"^vod" "symbols_^vod.png" 30;
"motata" "word_motata.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"x _ _ _ _ _" "word_motata_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xggtw" "consonants_xggtw.png" 20;
"gongi" "word_gongi.png" 10;
"fuksi" "word_fuksi.png" 10;
"d^dvdd" "symbols_d^dvdd.png" 30;
"d^oood" "symbols_d^oood.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _ _" "symbols_d^oood_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tvzdg" "consonants_tvzdg.png" 20;
"katodi" "word_katodi.png" 10;
"juhta" "word_juhta.png" 10;
"s^vds^" "symbols_s^vds^.png" 30;
"d^^do" "symbols_d^^do.png" 30;
"dvqdj" "consonants_dvqdj.png" 20;
"harjus" "word_harjus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"h _ _ _ _ _" "word_harjus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^v^" "symbols_v^v^.png" 30;
"akti" "word_akti.png" 10;
"päkiä" "word_päkiä.png" 10;
"kolhia" "word_kolhia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ i" "word_kolhia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^ss^" "symbols_^ss^.png" 30;
"fwgntw" "consonants_fwgntw.png" 20;
"orpo" "word_orpo.png" 10;
"ovvds" "symbols_ovvds.png" 30;
"miten" "word_miten.png" 10;
"zkwnr" "consonants_zkwnr.png" 20;
"sovo" "symbols_sovo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _" "symbols_sovo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dwzsrc" "consonants_dwzsrc.png" 20;
"poru" "word_poru.png" 10;
"kyhmy" "word_kyhmy.png" 10;
"ripsi" "word_ripsi.png" 10;
"dsovsd" "symbols_dsovsd.png" 30;
"häpy" "word_häpy.png" 10;
"zsgj" "consonants_zsgj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m" "consonants_zsgj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syaani" "word_syaani.png" 10;
"fbtsr" "consonants_fbtsr.png" 20;
"terska" "word_terska.png" 10;
"ssods" "symbols_ssods.png" 30;
"salaus" "word_salaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_salaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"almu" "word_almu.png" 10;
"hmff" "consonants_hmff.png" 20;
"riimu" "word_riimu.png" 10;
"tdgw" "consonants_tdgw.png" 20;
"rmmrh" "consonants_rmmrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_rmmrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vtkdtx" "consonants_vtkdtx.png" 20;
"hioa" "word_hioa.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"köli" "word_köli.png" 10;
"räntä" "word_räntä.png" 10;
"vdvv" "symbols_vdvv.png" 30;
"kolvi" "word_kolvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ü _ _ _" "word_kolvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"odote" "word_odote.png" 10;
"btmk" "consonants_btmk.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"kuje" "word_kuje.png" 10;
"lohi" "word_lohi.png" 10;
"lcrzjb" "consonants_lcrzjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ b _ _" "consonants_lcrzjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kyteä" "word_kyteä.png" 10;
"lxtgwm" "consonants_lxtgwm.png" 20;
"jänne" "word_jänne.png" 10;
"rwfcq" "consonants_rwfcq.png" 20;
"bjwqk" "consonants_bjwqk.png" 20;
"noppa" "word_noppa.png" 10;
"ovo^s" "symbols_ovo^s.png" 30;
"zdvv" "consonants_zdvv.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "consonants_zdvv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ilmi" "word_ilmi.png" 10;
"lemu" "word_lemu.png" 10;
"s^od" "symbols_s^od.png" 30;
"^osd" "symbols_^osd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_^osd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^s^s" "symbols_s^s^s.png" 30;
"mäntä" "word_mäntä.png" 10;
"tyköä" "word_tyköä.png" 10;
"gnbtn" "consonants_gnbtn.png" 20;
"xgtt" "consonants_xgtt.png" 20;
"ds^^o" "symbols_ds^^o.png" 30;
"qgjwzq" "consonants_qgjwzq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"w _ _ _ _ _" "consonants_qgjwzq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^vo" "symbols_s^vo.png" 30;
"nirso" "word_nirso.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"jaardi" "word_jaardi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ e" "word_jaardi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syöpyä" "word_syöpyä.png" 10;
"gchqcz" "consonants_gchqcz.png" 20;
"vo^^" "symbols_vo^^.png" 30;
"bwdz" "consonants_bwdz.png" 20;
"kuskus" "word_kuskus.png" 10;
"salvaa" "word_salvaa.png" 10;
"kphwjz" "consonants_kphwjz.png" 20;
"zpcqc" "consonants_zpcqc.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ c" "consonants_zpcqc_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kiuas" "word_kiuas.png" 10;
"vv^^" "symbols_vv^^.png" 30;
"oo^vdv" "symbols_oo^vdv.png" 30;
"nide" "word_nide.png" 10;
"xwnh" "consonants_xwnh.png" 20;
"vdso" "symbols_vdso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_vdso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vs^vds" "symbols_vs^vds.png" 30;
"mkfz" "consonants_mkfz.png" 20;
"svvd" "symbols_svvd.png" 30;
"rulla" "word_rulla.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _ _" "word_rulla_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"karies" "word_karies.png" 10;
"sopu" "word_sopu.png" 10;
"dsdoo^" "symbols_dsdoo^.png" 30;
"dssddd" "symbols_dssddd.png" 30;
"vaje" "word_vaje.png" 10;
"klaava" "word_klaava.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _ _ _" "word_klaava_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tykö" "word_tykö.png" 10;
"vovso" "symbols_vovso.png" 30;
"doosv" "symbols_doosv.png" 30;
"blkxwz" "consonants_blkxwz.png" 20;
"möly" "word_möly.png" 10;
"puida" "word_puida.png" 10;
"ovdvd" "symbols_ovdvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "symbols_ovdvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^s" "symbols_^s^s.png" 30;
"^d^^d" "symbols_^d^^d.png" 30;
"isyys" "word_isyys.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
"loraus" "word_loraus.png" 10;
"näkö" "word_näkö.png" 10;
"s^^d" "symbols_s^^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ ^ _ _" "symbols_s^^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"myyty" "word_myyty.png" 10;
"kaapia" "word_kaapia.png" 10;
"uuhi" "word_uuhi.png" 10;
"sodvdv" "symbols_sodvdv.png" 30;
"äänes" "word_äänes.png" 10;
"kihu" "word_kihu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_kihu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsd^ds" "symbols_vsd^ds.png" 30;
"tkcqd" "consonants_tkcqd.png" 20;
"stoola" "word_stoola.png" 10;
"vssd" "symbols_vssd.png" 30;
"iäti" "word_iäti.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"zfjxqk" "consonants_zfjxqk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _ _" "consonants_zfjxqk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vahaus" "word_vahaus.png" 10;
"^^^os" "symbols_^^^os.png" 30;
"älytä" "word_älytä.png" 10;
"dxfxv" "consonants_dxfxv.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _" "consonants_dxfxv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyrä" "word_tyrä.png" 10;
"itää" "word_itää.png" 10;
"bhsx" "consonants_bhsx.png" 20;
"^so^d" "symbols_^so^d.png" 30;
"sävy" "word_sävy.png" 10;
"druidi" "word_druidi.png" 10;
"dv^od" "symbols_dv^od.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "symbols_dv^od_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^vsov" "symbols_v^vsov.png" 30;
"tcfwdr" "consonants_tcfwdr.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"^sodvv" "symbols_^sodvv.png" 30;
"otsoni" "word_otsoni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ü _" "word_otsoni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvssv" "symbols_svvssv.png" 30;
"^ddv" "symbols_^ddv.png" 30;
"pyörö" "word_pyörö.png" 10;
"vuoka" "word_vuoka.png" 10;
"bzhkd" "consonants_bzhkd.png" 20;
"shiia" "word_shiia.png" 10;
"lhqt" "consonants_lhqt.png" 20;
"vovdsv" "symbols_vovdsv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ v" "symbols_vovdsv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rrwj" "consonants_rrwj.png" 20;
"sdvsv" "symbols_sdvsv.png" 30;
"känsä" "word_känsä.png" 10;
"sikhi" "word_sikhi.png" 10;
"hamaan" "word_hamaan.png" 10;
"spmb" "consonants_spmb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ m _" "consonants_spmb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"so^^v" "symbols_so^^v.png" 30;
"^vvs^s" "symbols_^vvs^s.png" 30;
"ripeä" "word_ripeä.png" 10;
"jdfjs" "consonants_jdfjs.png" 20;
"os^ood" "symbols_os^ood.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _ _ _" "symbols_os^ood_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tlzr" "consonants_tlzr.png" 20;
"nieriä" "word_nieriä.png" 10;
"kustos" "word_kustos.png" 10;
"estyä" "word_estyä.png" 10;
"kopina" "word_kopina.png" 10;
"s^dv^" "symbols_s^dv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_s^dv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"diodi" "word_diodi.png" 10;
"häkä" "word_häkä.png" 10;
"rukki" "word_rukki.png" 10;
"dovv^" "symbols_dovv^.png" 30;
"särkyä" "word_särkyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"j _ _ _ _ _" "word_särkyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kopio" "word_kopio.png" 10;
"täällä" "word_täällä.png" 10;
"voov^v" "symbols_voov^v.png" 30;
"voss" "symbols_voss.png" 30;
"^vd^vd" "symbols_^vd^vd.png" 30;
"tenä" "word_tenä.png" 10;
"säie" "word_säie.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_säie_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nuha" "word_nuha.png" 10;
"arpoa" "word_arpoa.png" 10;
"ositus" "word_ositus.png" 10;
"bxhl" "consonants_bxhl.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"b _ _ _" "consonants_bxhl_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kussa" "word_kussa.png" 10;
"psalmi" "word_psalmi.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"someen" "word_someen.png" 10;
"dvsv" "symbols_dvsv.png" 30;
"zztr" "consonants_zztr.png" 20;
"säiky" "word_säiky.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_säiky_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gxqtx" "consonants_gxqtx.png" 20;
"odo^" "symbols_odo^.png" 30;
"uuma" "word_uuma.png" 10;
"oosvsd" "symbols_oosvsd.png" 30;
"odvs^" "symbols_odvs^.png" 30;
"^svs^" "symbols_^svs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^svs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"yöpyä" "word_yöpyä.png" 10;
"^vovvs" "symbols_^vovvs.png" 30;
"hjqc" "consonants_hjqc.png" 20;
"jymy" "word_jymy.png" 10;
"pesin" "word_pesin.png" 10;
"kerubi" "word_kerubi.png" 10;
"suti" "word_suti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_suti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xkfvmb" "consonants_xkfvmb.png" 20;
"häät" "word_häät.png" 10;
"duuri" "word_duuri.png" 10;
"nurin" "word_nurin.png" 10;
"rfclk" "consonants_rfclk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ k" "consonants_rfclk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xxqhnz" "consonants_xxqhnz.png" 20;
"lakea" "word_lakea.png" 10;
"vdod^" "symbols_vdod^.png" 30;
"d^sd" "symbols_d^sd.png" 30;
"tämä" "word_tämä.png" 10;
"pmxwht" "consonants_pmxwht.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"p _ _ _ _ _" "consonants_pmxwht_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"suippo" "word_suippo.png" 10;
"lähi" "word_lähi.png" 10;
"opaali" "word_opaali.png" 10;
"osinko" "word_osinko.png" 10;
"silaus" "word_silaus.png" 10;
"uivelo" "word_uivelo.png" 10;
"do^d" "symbols_do^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s" "symbols_do^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pidot" "word_pidot.png" 10;
"ampuja" "word_ampuja.png" 10;
"uoma" "word_uoma.png" 10;
"mxnhh" "consonants_mxnhh.png" 20;
"uuni" "word_uuni.png" 10;
"uima" "word_uima.png" 10;
"silsa" "word_silsa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ p _ _ _" "word_silsa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdosd" "symbols_sdosd.png" 30;
"ajos" "word_ajos.png" 10;
"krvvpr" "consonants_krvvpr.png" 20;
"dqkcl" "consonants_dqkcl.png" 20;
"lempo" "word_lempo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _ _" "word_lempo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uute" "word_uute.png" 10;
"tczhj" "consonants_tczhj.png" 20;
"äimä" "word_äimä.png" 10;
"ztlrlf" "consonants_ztlrlf.png" 20;
"kohu" "word_kohu.png" 10;
"pesula" "word_pesula.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_pesula_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bfkx" "consonants_bfkx.png" 20;
"urut" "word_urut.png" 10;
"räme" "word_räme.png" 10;
"whpw" "consonants_whpw.png" 20;
"jqdsmj" "consonants_jqdsmj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n _" "consonants_jqdsmj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rjfcdx" "consonants_rjfcdx.png" 20;
"s^od^" "symbols_s^od^.png" 30;
"erkani" "word_erkani.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
"näppy" "word_näppy.png" 10;
"seimi" "word_seimi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m _" "word_seimi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sv^vvs" "symbols_sv^vvs.png" 30;
"hxjlgq" "consonants_hxjlgq.png" 20;
"korren" "word_korren.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
"xpfpj" "consonants_xpfpj.png" 20;
"sarake" "word_sarake.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"ä _ _ _ _ _" "word_sarake_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jbntmc" "consonants_jbntmc.png" 20;
"säkä" "word_säkä.png" 10;
"^dso" "symbols_^dso.png" 30;
"txhs" "consonants_txhs.png" 20;
"lblxm" "consonants_lblxm.png" 20;
"dovdd" "symbols_dovdd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_dovdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gljt" "consonants_gljt.png" 20;
"oinas" "word_oinas.png" 10;
"vsd^" "symbols_vsd^.png" 30;
"qckq" "consonants_qckq.png" 20;
"jxfm" "consonants_jxfm.png" 20;
"huorin" "word_huorin.png" 10;
"tuohus" "word_tuohus.png" 10;
"eliö" "word_eliö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _" "word_eliö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"laukka" "word_laukka.png" 10;
"ässä" "word_ässä.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ z" "consonants_ksrwvk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fotoni" "word_fotoni.png" 10;
"säle" "word_säle.png" 10;
"vwrptk" "consonants_vwrptk.png" 20;
"^oddo" "symbols_^oddo.png" 30;
"rapsi" "word_rapsi.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"menijä" "word_menijä.png" 10;
"dlgt" "consonants_dlgt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "consonants_dlgt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvo^" "symbols_vvo^.png" 30;
"nurja" "word_nurja.png" 10;
"itiö" "word_itiö.png" 10;
"^^od" "symbols_^^od.png" 30;
"vzkt" "consonants_vzkt.png" 20;
"gsdx" "consonants_gsdx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _" "consonants_gsdx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"siitos" "word_siitos.png" 10;
"osvs^d" "symbols_osvs^d.png" 30;
"vs^oo" "symbols_vs^oo.png" 30;
"grgdd" "consonants_grgdd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _" "consonants_grgdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dfszzp" "consonants_dfszzp.png" 20;
"lotja" "word_lotja.png" 10;
"ahjo" "word_ahjo.png" 10;
"ratamo" "word_ratamo.png" 10;
"gthb" "consonants_gthb.png" 20;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ähky" "word_ähky.png" 10;
"wnnz" "consonants_wnnz.png" 20;
"ääriin" "word_ääriin.png" 10;
"s^o^^v" "symbols_s^o^^v.png" 30;
"czkdbs" "consonants_czkdbs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"c _ _ _ _ _" "consonants_czkdbs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"luusto" "word_luusto.png" 10;
"vvvd" "symbols_vvvd.png" 30;
"uiva" "word_uiva.png" 10;
"vipu" "word_vipu.png" 10;
"osdod^" "symbols_osdod^.png" 30;
"tyvi" "word_tyvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _" "word_tyvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdvdvd" "symbols_sdvdvd.png" 30;
"äänne" "word_äänne.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"hajan" "word_hajan.png" 10;
"hihna" "word_hihna.png" 10;
"kytkin" "word_kytkin.png" 10;
"torium" "word_torium.png" 10;
"pamppu" "word_pamppu.png" 10;
"emätin" "word_emätin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"c _ _ _ _ _" "word_emätin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qwjvhz" "consonants_qwjvhz.png" 20;
"d^v^^" "symbols_d^v^^.png" 30;
"odos" "symbols_odos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _" "symbols_odos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"voo^o" "symbols_voo^o.png" 30;
"piip" "word_piip.png" 10;
"jaos" "word_jaos.png" 10;
"sysi" "word_sysi.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"pöty" "word_pöty.png" 10;
"rrggj" "consonants_rrggj.png" 20;
"nisä" "word_nisä.png" 10;
"solmio" "word_solmio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_solmio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qpvbs" "consonants_qpvbs.png" 20;
"hautua" "word_hautua.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"dvsvos" "symbols_dvsvos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ s _ _ _" "symbols_dvsvos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jakaja" "word_jakaja.png" 10;
"kortti" "word_kortti.png" 10;
"mwgz" "consonants_mwgz.png" 20;
"afasia" "word_afasia.png" 10;
"faksi" "word_faksi.png" 10;
"ltqrr" "consonants_ltqrr.png" 20;
"kaste" "word_kaste.png" 10;
"pitäen" "word_pitäen.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_pitäen_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lgtjb" "consonants_lgtjb.png" 20;
"pokeri" "word_pokeri.png" 10;
"ssss" "symbols_ssss.png" 30;
"o^ss" "symbols_o^ss.png" 30;
"jäte" "word_jäte.png" 10;
"kysta" "word_kysta.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_kysta_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
};
