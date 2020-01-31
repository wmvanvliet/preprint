
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
"sarake" "word_sarake.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"tenä" "word_tenä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _" "word_tenä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyrä" "word_tyrä.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"tykö" "word_tykö.png" 10;
"pmxwht" "consonants_pmxwht.png" 20;
"mgttwx" "consonants_mgttwx.png" 20;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dssddd" "symbols_dssddd.png" 30;
"krvvpr" "consonants_krvvpr.png" 20;
"doosv" "symbols_doosv.png" 30;
"piip" "word_piip.png" 10;
"näppy" "word_näppy.png" 10;
"someen" "word_someen.png" 10;
"voss" "symbols_voss.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s" "symbols_voss_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äänes" "word_äänes.png" 10;
"viipyä" "word_viipyä.png" 10;
"ositus" "word_ositus.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"kyteä" "word_kyteä.png" 10;
"hjqc" "consonants_hjqc.png" 20;
"jtltfj" "consonants_jtltfj.png" 20;
"afasia" "word_afasia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ z" "word_afasia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"harjus" "word_harjus.png" 10;
"jaardi" "word_jaardi.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^d^vs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^osd" "symbols_^osd.png" 30;
"oosvsd" "symbols_oosvsd.png" 30;
"bwdz" "consonants_bwdz.png" 20;
"klaava" "word_klaava.png" 10;
"fotoni" "word_fotoni.png" 10;
"gchqcz" "consonants_gchqcz.png" 20;
"^^svd" "symbols_^^svd.png" 30;
"ssss" "symbols_ssss.png" 30;
"rjfcdx" "consonants_rjfcdx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ x _" "consonants_rjfcdx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xgtt" "consonants_xgtt.png" 20;
"vovvvv" "symbols_vovvvv.png" 30;
"qwjvhz" "consonants_qwjvhz.png" 20;
"jakaja" "word_jakaja.png" 10;
"so^^v" "symbols_so^^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^ _" "symbols_so^^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sävy" "word_sävy.png" 10;
"ltqrr" "consonants_ltqrr.png" 20;
"vssd" "symbols_vssd.png" 30;
"täällä" "word_täällä.png" 10;
"ähky" "word_ähky.png" 10;
"bjwqk" "consonants_bjwqk.png" 20;
"kihu" "word_kihu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_kihu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jxfm" "consonants_jxfm.png" 20;
"tdgw" "consonants_tdgw.png" 20;
"nuha" "word_nuha.png" 10;
"tcftcg" "consonants_tcftcg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ c _ _ _ _" "consonants_tcftcg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"räme" "word_räme.png" 10;
"lotja" "word_lotja.png" 10;
"dvsvos" "symbols_dvsvos.png" 30;
"^s^s" "symbols_^s^s.png" 30;
"qgjwzq" "consonants_qgjwzq.png" 20;
"vo^^" "symbols_vo^^.png" 30;
"stoola" "word_stoola.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ t _ _ _ _" "word_stoola_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lhqt" "consonants_lhqt.png" 20;
"dfszzp" "consonants_dfszzp.png" 20;
"kortti" "word_kortti.png" 10;
"druidi" "word_druidi.png" 10;
"psalmi" "word_psalmi.png" 10;
"kphwjz" "consonants_kphwjz.png" 20;
"säle" "word_säle.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "word_säle_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuma" "word_uuma.png" 10;
"vovdsv" "symbols_vovdsv.png" 30;
"ilmi" "word_ilmi.png" 10;
"uoma" "word_uoma.png" 10;
"nisä" "word_nisä.png" 10;
"karies" "word_karies.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ i _ _" "word_karies_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tczhj" "consonants_tczhj.png" 20;
"sdvdvd" "symbols_sdvdvd.png" 30;
"orpo" "word_orpo.png" 10;
"sysi" "word_sysi.png" 10;
"pitäen" "word_pitäen.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_pitäen_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^d^^d" "symbols_^d^^d.png" 30;
"särkyä" "word_särkyä.png" 10;
"lgtjb" "consonants_lgtjb.png" 20;
"urut" "word_urut.png" 10;
"^^od" "symbols_^^od.png" 30;
"vovso" "symbols_vovso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "symbols_vovso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rapsi" "word_rapsi.png" 10;
"erkani" "word_erkani.png" 10;
"lemu" "word_lemu.png" 10;
"v^v^" "symbols_v^v^.png" 30;
"zztr" "consonants_zztr.png" 20;
"gljt" "consonants_gljt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t" "consonants_gljt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dvsv" "symbols_dvsv.png" 30;
"mäntä" "word_mäntä.png" 10;
"s^vds^" "symbols_s^vds^.png" 30;
"emätin" "word_emätin.png" 10;
"whpw" "consonants_whpw.png" 20;
"vsd^ds" "symbols_vsd^ds.png" 30;
"jymy" "word_jymy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_jymy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuhi" "word_uuhi.png" 10;
"tyköä" "word_tyköä.png" 10;
"kussa" "word_kussa.png" 10;
"ripsi" "word_ripsi.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jqdsmj" "consonants_jqdsmj.png" 20;
"kohu" "word_kohu.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"d^dvdd" "symbols_d^dvdd.png" 30;
"kopio" "word_kopio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o" "word_kopio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"menijä" "word_menijä.png" 10;
"terska" "word_terska.png" 10;
"voo^o" "symbols_voo^o.png" 30;
"gsdx" "consonants_gsdx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _" "consonants_gsdx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mxnhh" "consonants_mxnhh.png" 20;
"pesin" "word_pesin.png" 10;
"myyty" "word_myyty.png" 10;
"dwzsrc" "consonants_dwzsrc.png" 20;
"^^^os" "symbols_^^^os.png" 30;
"diodi" "word_diodi.png" 10;
"lxtgwm" "consonants_lxtgwm.png" 20;
"kiuas" "word_kiuas.png" 10;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gongi" "word_gongi.png" 10;
"kaapia" "word_kaapia.png" 10;
"tlzr" "consonants_tlzr.png" 20;
"rrggj" "consonants_rrggj.png" 20;
"rwfcq" "consonants_rwfcq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _ _" "consonants_rwfcq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^sodvv" "symbols_^sodvv.png" 30;
"hihna" "word_hihna.png" 10;
"gthb" "consonants_gthb.png" 20;
"bzhkd" "consonants_bzhkd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ l _ _ _" "consonants_bzhkd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häkä" "word_häkä.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"korren" "word_korren.png" 10;
"tcfwdr" "consonants_tcfwdr.png" 20;
"vvvd" "symbols_vvvd.png" 30;
"odos" "symbols_odos.png" 30;
"kaste" "word_kaste.png" 10;
"köli" "word_köli.png" 10;
"silaus" "word_silaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "word_silaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ripeä" "word_ripeä.png" 10;
"s^o^^v" "symbols_s^o^^v.png" 30;
"juhta" "word_juhta.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "word_juhta_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tvzdg" "consonants_tvzdg.png" 20;
"s^dv^" "symbols_s^dv^.png" 30;
"itiö" "word_itiö.png" 10;
"tkcqd" "consonants_tkcqd.png" 20;
"dfdzj" "consonants_dfdzj.png" 20;
"ovo^s" "symbols_ovo^s.png" 30;
"itää" "word_itää.png" 10;
"blkxwz" "consonants_blkxwz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ w _" "consonants_blkxwz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"voov^v" "symbols_voov^v.png" 30;
"xxqhnz" "consonants_xxqhnz.png" 20;
"oo^vdv" "symbols_oo^vdv.png" 30;
"ztlrlf" "consonants_ztlrlf.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ c _ _" "consonants_ztlrlf_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"odvs^" "symbols_odvs^.png" 30;
"tyvi" "word_tyvi.png" 10;
"dv^od" "symbols_dv^od.png" 30;
"vuoka" "word_vuoka.png" 10;
"tuohus" "word_tuohus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ u _ _" "word_tuohus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pidot" "word_pidot.png" 10;
"vs^oo" "symbols_vs^oo.png" 30;
"sv^vvs" "symbols_sv^vvs.png" 30;
"kopina" "word_kopina.png" 10;
"vvo^" "symbols_vvo^.png" 30;
"jdfjs" "consonants_jdfjs.png" 20;
"^vovvs" "symbols_^vovvs.png" 30;
"d^oood" "symbols_d^oood.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _ _" "symbols_d^oood_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^^d" "symbols_s^^d.png" 30;
"sodvdv" "symbols_sodvdv.png" 30;
"^dso" "symbols_^dso.png" 30;
"shiia" "word_shiia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_shiia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdvsv" "symbols_sdvsv.png" 30;
"^oddo" "symbols_^oddo.png" 30;
"zdvv" "consonants_zdvv.png" 20;
"s^od" "symbols_s^od.png" 30;
"kytkin" "word_kytkin.png" 10;
"ylkä" "word_ylkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "word_ylkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hautua" "word_hautua.png" 10;
"osdod^" "symbols_osdod^.png" 30;
"uute" "word_uute.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"^ss^" "symbols_^ss^.png" 30;
"gxqtx" "consonants_gxqtx.png" 20;
"vv^^" "symbols_vv^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _" "symbols_vv^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^sd" "symbols_d^sd.png" 30;
"mkfz" "consonants_mkfz.png" 20;
"gnbtn" "consonants_gnbtn.png" 20;
"säkä" "word_säkä.png" 10;
"jänne" "word_jänne.png" 10;
"wnnz" "consonants_wnnz.png" 20;
"ässä" "word_ässä.png" 10;
"seimi" "word_seimi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m _" "word_seimi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bhsx" "consonants_bhsx.png" 20;
"qvkjt" "consonants_qvkjt.png" 20;
"suippo" "word_suippo.png" 10;
"txhs" "consonants_txhs.png" 20;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kysta" "word_kysta.png" 10;
"almu" "word_almu.png" 10;
"räntä" "word_räntä.png" 10;
"fuksi" "word_fuksi.png" 10;
"känsä" "word_känsä.png" 10;
"lohi" "word_lohi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ q _" "word_lohi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säie" "word_säie.png" 10;
"rukki" "word_rukki.png" 10;
"ssods" "symbols_ssods.png" 30;
"jäte" "word_jäte.png" 10;
"nirso" "word_nirso.png" 10;
"älytä" "word_älytä.png" 10;
"vzkt" "consonants_vzkt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "consonants_vzkt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kuje" "word_kuje.png" 10;
"dqkcl" "consonants_dqkcl.png" 20;
"miten" "word_miten.png" 10;
"häät" "word_häät.png" 10;
"salvaa" "word_salvaa.png" 10;
"syaani" "word_syaani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "word_syaani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dsdoo^" "symbols_dsdoo^.png" 30;
"osuma" "word_osuma.png" 10;
"eliö" "word_eliö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _" "word_eliö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jbntmc" "consonants_jbntmc.png" 20;
"arpoa" "word_arpoa.png" 10;
"dlgt" "consonants_dlgt.png" 20;
"odo^" "symbols_odo^.png" 30;
"lcrzjb" "consonants_lcrzjb.png" 20;
"ratamo" "word_ratamo.png" 10;
"opaali" "word_opaali.png" 10;
"zsgj" "consonants_zsgj.png" 20;
"rrwj" "consonants_rrwj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _" "consonants_rrwj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"torium" "word_torium.png" 10;
"salaus" "word_salaus.png" 10;
"vahaus" "word_vahaus.png" 10;
"lakea" "word_lakea.png" 10;
"ääriin" "word_ääriin.png" 10;
"otsoni" "word_otsoni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ü _" "word_otsoni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^s^s" "symbols_s^s^s.png" 30;
"vdso" "symbols_vdso.png" 30;
"vipu" "word_vipu.png" 10;
"ajos" "word_ajos.png" 10;
"s^od^" "symbols_s^od^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _ _" "symbols_s^od^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jaos" "word_jaos.png" 10;
"xwnh" "consonants_xwnh.png" 20;
"xpfpj" "consonants_xpfpj.png" 20;
"häpy" "word_häpy.png" 10;
"pokeri" "word_pokeri.png" 10;
"^svs^" "symbols_^svs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^svs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fwgntw" "consonants_fwgntw.png" 20;
"duuri" "word_duuri.png" 10;
"kolvi" "word_kolvi.png" 10;
"nurja" "word_nurja.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"d^v^^" "symbols_d^v^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ ^ _ _ _" "symbols_d^v^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"wcjrjq" "consonants_wcjrjq.png" 20;
"odote" "word_odote.png" 10;
"ovvds" "symbols_ovvds.png" 30;
"^d^oos" "symbols_^d^oos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s _ _" "symbols_^d^oos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"akti" "word_akti.png" 10;
"isyys" "word_isyys.png" 10;
"vwrptk" "consonants_vwrptk.png" 20;
"uuni" "word_uuni.png" 10;
"fbtsr" "consonants_fbtsr.png" 20;
"rmmrh" "consonants_rmmrh.png" 20;
"sovo" "symbols_sovo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _" "symbols_sovo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zkwnr" "consonants_zkwnr.png" 20;
"luusto" "word_luusto.png" 10;
"kuskus" "word_kuskus.png" 10;
"ampuja" "word_ampuja.png" 10;
"mwgz" "consonants_mwgz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _" "consonants_mwgz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"iäti" "word_iäti.png" 10;
"svvssv" "symbols_svvssv.png" 30;
"xggtw" "consonants_xggtw.png" 20;
"svvd" "symbols_svvd.png" 30;
"dovdd" "symbols_dovdd.png" 30;
"lblxm" "consonants_lblxm.png" 20;
"qckq" "consonants_qckq.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kolhia" "word_kolhia.png" 10;
"oinas" "word_oinas.png" 10;
"vaje" "word_vaje.png" 10;
"suti" "word_suti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_suti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dsovsd" "symbols_dsovsd.png" 30;
"os^ood" "symbols_os^ood.png" 30;
"päkiä" "word_päkiä.png" 10;
"hmff" "consonants_hmff.png" 20;
"puida" "word_puida.png" 10;
"qpvbs" "consonants_qpvbs.png" 20;
"^ddv" "symbols_^ddv.png" 30;
"bxhl" "consonants_bxhl.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"b _ _ _" "consonants_bxhl_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uivelo" "word_uivelo.png" 10;
"pöty" "word_pöty.png" 10;
"säiky" "word_säiky.png" 10;
"riimu" "word_riimu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_riimu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"faksi" "word_faksi.png" 10;
"osvs^d" "symbols_osvs^d.png" 30;
"ovdvd" "symbols_ovdvd.png" 30;
"lempo" "word_lempo.png" 10;
"vtkdtx" "consonants_vtkdtx.png" 20;
"vdod^" "symbols_vdod^.png" 30;
"huorin" "word_huorin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _ _" "word_huorin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zfjxqk" "consonants_zfjxqk.png" 20;
"nieriä" "word_nieriä.png" 10;
"rulla" "word_rulla.png" 10;
"pyörö" "word_pyörö.png" 10;
"ahjo" "word_ahjo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ h _ _" "word_ahjo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"noppa" "word_noppa.png" 10;
"estyä" "word_estyä.png" 10;
"siitos" "word_siitos.png" 10;
"poru" "word_poru.png" 10;
"vs^vds" "symbols_vs^vds.png" 30;
"hamaan" "word_hamaan.png" 10;
"d^^do" "symbols_d^^do.png" 30;
"^so^d" "symbols_^so^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_^so^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^vo" "symbols_s^vo.png" 30;
"^vvs^s" "symbols_^vvs^s.png" 30;
"hioa" "word_hioa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "word_hioa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vod" "symbols_^vod.png" 30;
"xkfvmb" "consonants_xkfvmb.png" 20;
"czkdbs" "consonants_czkdbs.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"äänne" "word_äänne.png" 10;
"uima" "word_uima.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_uima_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äimä" "word_äimä.png" 10;
"motata" "word_motata.png" 10;
"dvqdj" "consonants_dvqdj.png" 20;
"ds^^o" "symbols_ds^^o.png" 30;
"näkö" "word_näkö.png" 10;
"hxjlgq" "consonants_hxjlgq.png" 20;
"sdosd" "symbols_sdosd.png" 30;
"o^ss" "symbols_o^ss.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s" "symbols_o^ss_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bfkx" "consonants_bfkx.png" 20;
"laukka" "word_laukka.png" 10;
"grgdd" "consonants_grgdd.png" 20;
"uiva" "word_uiva.png" 10;
"nurin" "word_nurin.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
"loraus" "word_loraus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ o _ _ _ _" "word_loraus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesula" "word_pesula.png" 10;
"nide" "word_nide.png" 10;
"v^vsov" "symbols_v^vsov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_v^vsov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kyhmy" "word_kyhmy.png" 10;
"rämä" "word_rämä.png" 10;
"sikhi" "word_sikhi.png" 10;
"silsa" "word_silsa.png" 10;
"möly" "word_möly.png" 10;
"katodi" "word_katodi.png" 10;
"do^d" "symbols_do^d.png" 30;
"rfclk" "consonants_rfclk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ k" "consonants_rfclk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lähi" "word_lähi.png" 10;
"vsd^" "symbols_vsd^.png" 30;
"dovv^" "symbols_dovv^.png" 30;
"solmio" "word_solmio.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v _" "consonants_ndsvw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dxfxv" "consonants_dxfxv.png" 20;
"sopu" "word_sopu.png" 10;
"tämä" "word_tämä.png" 10;
"btmk" "consonants_btmk.png" 20;
"osinko" "word_osinko.png" 10;
"kerubi" "word_kerubi.png" 10;
"spmb" "consonants_spmb.png" 20;
"vdvv" "symbols_vdvv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_vdvv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
};
