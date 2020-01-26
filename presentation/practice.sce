
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
"txhs" "consonants_txhs.png" 20;
"svvd" "symbols_svvd.png" 30;
"rfclk" "consonants_rfclk.png" 20;
"syaani" "word_syaani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "word_syaani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"siitos" "word_siitos.png" 10;
"odos" "symbols_odos.png" 30;
"vv^^" "symbols_vv^^.png" 30;
"d^v^^" "symbols_d^v^^.png" 30;
"uiva" "word_uiva.png" 10;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oinas" "word_oinas.png" 10;
"estyä" "word_estyä.png" 10;
"hormi" "word_hormi.png" 10;
"uivelo" "word_uivelo.png" 10;
"hjqc" "consonants_hjqc.png" 20;
"^svs^" "symbols_^svs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^svs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsd^ds" "symbols_vsd^ds.png" 30;
"bjwqk" "consonants_bjwqk.png" 20;
"ripsi" "word_ripsi.png" 10;
"hmff" "consonants_hmff.png" 20;
"mkfz" "consonants_mkfz.png" 20;
"lemu" "word_lemu.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesula" "word_pesula.png" 10;
"juhta" "word_juhta.png" 10;
"nisä" "word_nisä.png" 10;
"ds^^o" "symbols_ds^^o.png" 30;
"sv^vvs" "symbols_sv^vvs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o _" "symbols_sv^vvs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"luusto" "word_luusto.png" 10;
"eliö" "word_eliö.png" 10;
"sysi" "word_sysi.png" 10;
"voo^o" "symbols_voo^o.png" 30;
"rapsi" "word_rapsi.png" 10;
"tuohus" "word_tuohus.png" 10;
"xpfpj" "consonants_xpfpj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ p _ _ _" "consonants_xpfpj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"piip" "word_piip.png" 10;
"dvsvos" "symbols_dvsvos.png" 30;
"jbntmc" "consonants_jbntmc.png" 20;
"gchqcz" "consonants_gchqcz.png" 20;
"ähky" "word_ähky.png" 10;
"sdvsv" "symbols_sdvsv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _ _" "symbols_sdvsv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äänes" "word_äänes.png" 10;
"fotoni" "word_fotoni.png" 10;
"wcjrjq" "consonants_wcjrjq.png" 20;
"säie" "word_säie.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_säie_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uoma" "word_uoma.png" 10;
"uima" "word_uima.png" 10;
"älytä" "word_älytä.png" 10;
"sikhi" "word_sikhi.png" 10;
"oosvsd" "symbols_oosvsd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _ _" "symbols_oosvsd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"loraus" "word_loraus.png" 10;
"^vovvs" "symbols_^vovvs.png" 30;
"nieriä" "word_nieriä.png" 10;
"^dso" "symbols_^dso.png" 30;
"tkcqd" "consonants_tkcqd.png" 20;
"zdvv" "consonants_zdvv.png" 20;
"ovo^s" "symbols_ovo^s.png" 30;
"tyköä" "word_tyköä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_tyköä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gthb" "consonants_gthb.png" 20;
"oo^v" "symbols_oo^v.png" 30;
"mäntä" "word_mäntä.png" 10;
"ilmi" "word_ilmi.png" 10;
"jqdsmj" "consonants_jqdsmj.png" 20;
"noppa" "word_noppa.png" 10;
"uuni" "word_uuni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"n _ _ _" "word_uuni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"karies" "word_karies.png" 10;
"nurin" "word_nurin.png" 10;
"silsa" "word_silsa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ p _ _ _" "word_silsa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^od" "symbols_s^od.png" 30;
"pidot" "word_pidot.png" 10;
"erkani" "word_erkani.png" 10;
"motata" "word_motata.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"sarake" "word_sarake.png" 10;
"opaali" "word_opaali.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"o _ _ _ _ _" "word_opaali_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ratamo" "word_ratamo.png" 10;
"iäti" "word_iäti.png" 10;
"vtkdtx" "consonants_vtkdtx.png" 20;
"spmb" "consonants_spmb.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"känsä" "word_känsä.png" 10;
"ssss" "symbols_ssss.png" 30;
"kortti" "word_kortti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t _ _" "word_kortti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"os^ood" "symbols_os^ood.png" 30;
"^vvs^s" "symbols_^vvs^s.png" 30;
"vahaus" "word_vahaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ s" "word_vahaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"voss" "symbols_voss.png" 30;
"lohi" "word_lohi.png" 10;
"jymy" "word_jymy.png" 10;
"pöty" "word_pöty.png" 10;
"ahjo" "word_ahjo.png" 10;
"druidi" "word_druidi.png" 10;
"kiuas" "word_kiuas.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "word_kiuas_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ttjglk" "consonants_ttjglk.png" 20;
"osuma" "word_osuma.png" 10;
"menijä" "word_menijä.png" 10;
"bwdz" "consonants_bwdz.png" 20;
"grgdd" "consonants_grgdd.png" 20;
"dssddd" "symbols_dssddd.png" 30;
"^d^^d" "symbols_^d^^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^d^^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tvzdg" "consonants_tvzdg.png" 20;
"sqdq" "consonants_sqdq.png" 20;
"ajos" "word_ajos.png" 10;
"ovdvd" "symbols_ovdvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "symbols_ovdvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kustos" "word_kustos.png" 10;
"osvs^d" "symbols_osvs^d.png" 30;
"häpy" "word_häpy.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"myyty" "word_myyty.png" 10;
"ampuja" "word_ampuja.png" 10;
"näppy" "word_näppy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _" "word_näppy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"miten" "word_miten.png" 10;
"isyys" "word_isyys.png" 10;
"s^s^s" "symbols_s^s^s.png" 30;
"jaardi" "word_jaardi.png" 10;
"tcftcg" "consonants_tcftcg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ c _ _ _ _" "consonants_tcftcg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^osd" "symbols_^osd.png" 30;
"pitäen" "word_pitäen.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"riimu" "word_riimu.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"vvosod" "symbols_vvosod.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _ _" "symbols_vvosod_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"diodi" "word_diodi.png" 10;
"kopio" "word_kopio.png" 10;
"salaus" "word_salaus.png" 10;
"tdgw" "consonants_tdgw.png" 20;
"ndsvw" "consonants_ndsvw.png" 20;
"jäte" "word_jäte.png" 10;
"ztlrlf" "consonants_ztlrlf.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ c _ _" "consonants_ztlrlf_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lgtjb" "consonants_lgtjb.png" 20;
"arpoa" "word_arpoa.png" 10;
"fbtsr" "consonants_fbtsr.png" 20;
"fwgntw" "consonants_fwgntw.png" 20;
"pesin" "word_pesin.png" 10;
"mgttwx" "consonants_mgttwx.png" 20;
"vo^^" "symbols_vo^^.png" 30;
"gnbtn" "consonants_gnbtn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _" "consonants_gnbtn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kphwjz" "consonants_kphwjz.png" 20;
"jxfm" "consonants_jxfm.png" 20;
"wnnz" "consonants_wnnz.png" 20;
"tcfwdr" "consonants_tcfwdr.png" 20;
"jänne" "word_jänne.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ p _" "word_jänne_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jakaja" "word_jakaja.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"rrggj" "consonants_rrggj.png" 20;
"so^^v" "symbols_so^^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^ _" "symbols_so^^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"seimi" "word_seimi.png" 10;
"^s^s" "symbols_^s^s.png" 30;
"lähi" "word_lähi.png" 10;
"vdso" "symbols_vdso.png" 30;
"s^o^^v" "symbols_s^o^^v.png" 30;
"ääriin" "word_ääriin.png" 10;
"häät" "word_häät.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ l" "word_häät_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vaje" "word_vaje.png" 10;
"jaos" "word_jaos.png" 10;
"terska" "word_terska.png" 10;
"blkxwz" "consonants_blkxwz.png" 20;
"itää" "word_itää.png" 10;
"uute" "word_uute.png" 10;
"vdvv" "symbols_vdvv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _" "symbols_vdvv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häkä" "word_häkä.png" 10;
"dv^od" "symbols_dv^od.png" 30;
"kopina" "word_kopina.png" 10;
"mxnhh" "consonants_mxnhh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ z" "consonants_mxnhh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"someen" "word_someen.png" 10;
"orpo" "word_orpo.png" 10;
"tämä" "word_tämä.png" 10;
"kaste" "word_kaste.png" 10;
"emätin" "word_emätin.png" 10;
"rukki" "word_rukki.png" 10;
"faksi" "word_faksi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _ _" "word_faksi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vs^oo" "symbols_vs^oo.png" 30;
"pokeri" "word_pokeri.png" 10;
"rämä" "word_rämä.png" 10;
"möly" "word_möly.png" 10;
"osdod^" "symbols_osdod^.png" 30;
"dsovsd" "symbols_dsovsd.png" 30;
"rmmrh" "consonants_rmmrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_rmmrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyvi" "word_tyvi.png" 10;
"osinko" "word_osinko.png" 10;
"qpvbs" "consonants_qpvbs.png" 20;
"äänne" "word_äänne.png" 10;
"solmio" "word_solmio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_solmio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xwnh" "consonants_xwnh.png" 20;
"puida" "word_puida.png" 10;
"gljt" "consonants_gljt.png" 20;
"vovvvv" "symbols_vovvvv.png" 30;
"urut" "word_urut.png" 10;
"särkyä" "word_särkyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"j _ _ _ _ _" "word_särkyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"doosv" "symbols_doosv.png" 30;
"vwrptk" "consonants_vwrptk.png" 20;
"shiia" "word_shiia.png" 10;
"ripeä" "word_ripeä.png" 10;
"v^v^" "symbols_v^v^.png" 30;
"kussa" "word_kussa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_kussa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"laukka" "word_laukka.png" 10;
"lempo" "word_lempo.png" 10;
"fuksi" "word_fuksi.png" 10;
"zkwnr" "consonants_zkwnr.png" 20;
"s^^d" "symbols_s^^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ ^ _ _" "symbols_s^^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vovso" "symbols_vovso.png" 30;
"dovv^" "symbols_dovv^.png" 30;
"pyörö" "word_pyörö.png" 10;
"psalmi" "word_psalmi.png" 10;
"^ddv" "symbols_^ddv.png" 30;
"harjus" "word_harjus.png" 10;
"säkä" "word_säkä.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"rrwj" "consonants_rrwj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _" "consonants_rrwj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^^^os" "symbols_^^^os.png" 30;
"lotja" "word_lotja.png" 10;
"vs^vds" "symbols_vs^vds.png" 30;
"xggtw" "consonants_xggtw.png" 20;
"odote" "word_odote.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ j" "word_odote_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nirso" "word_nirso.png" 10;
"itiö" "word_itiö.png" 10;
"kytkin" "word_kytkin.png" 10;
"suti" "word_suti.png" 10;
"whpw" "consonants_whpw.png" 20;
"dfdzj" "consonants_dfdzj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ j" "consonants_dfdzj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dovdd" "symbols_dovdd.png" 30;
"kysta" "word_kysta.png" 10;
"sdosd" "symbols_sdosd.png" 30;
"silaus" "word_silaus.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"akti" "word_akti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _" "word_akti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äimä" "word_äimä.png" 10;
"d^dvdd" "symbols_d^dvdd.png" 30;
"räme" "word_räme.png" 10;
"vuoka" "word_vuoka.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "word_vuoka_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"btmk" "consonants_btmk.png" 20;
"köli" "word_köli.png" 10;
"hxjlgq" "consonants_hxjlgq.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"voov^v" "symbols_voov^v.png" 30;
"d^sd" "symbols_d^sd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_d^sd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hihna" "word_hihna.png" 10;
"lhqt" "consonants_lhqt.png" 20;
"qckq" "consonants_qckq.png" 20;
"kolhia" "word_kolhia.png" 10;
"näkö" "word_näkö.png" 10;
"s^od^" "symbols_s^od^.png" 30;
"kaapia" "word_kaapia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"k _ _ _ _ _" "word_kaapia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kohu" "word_kohu.png" 10;
"ässä" "word_ässä.png" 10;
"odvs^" "symbols_odvs^.png" 30;
"sodvdv" "symbols_sodvdv.png" 30;
"hioa" "word_hioa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "word_hioa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"duuri" "word_duuri.png" 10;
"s^vds^" "symbols_s^vds^.png" 30;
"kerubi" "word_kerubi.png" 10;
"dxfxv" "consonants_dxfxv.png" 20;
"dwzsrc" "consonants_dwzsrc.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
"tenä" "word_tenä.png" 10;
"dsdoo^" "symbols_dsdoo^.png" 30;
"dvqdj" "consonants_dvqdj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ j" "consonants_dvqdj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xkfvmb" "consonants_xkfvmb.png" 20;
"pmxwht" "consonants_pmxwht.png" 20;
"lxtgwm" "consonants_lxtgwm.png" 20;
"bxhl" "consonants_bxhl.png" 20;
"nide" "word_nide.png" 10;
"kihu" "word_kihu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_kihu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^oddo" "symbols_^oddo.png" 30;
"vvo^" "symbols_vvo^.png" 30;
"rulla" "word_rulla.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _ _" "word_rulla_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qgjwzq" "consonants_qgjwzq.png" 20;
"tykö" "word_tykö.png" 10;
"dfszzp" "consonants_dfszzp.png" 20;
"klaava" "word_klaava.png" 10;
"^^svd" "symbols_^^svd.png" 30;
"viipyä" "word_viipyä.png" 10;
"tczhj" "consonants_tczhj.png" 20;
"lcrzjb" "consonants_lcrzjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ b _ _" "consonants_lcrzjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xxqhnz" "consonants_xxqhnz.png" 20;
"huorin" "word_huorin.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"säle" "word_säle.png" 10;
"hamaan" "word_hamaan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä _" "word_hamaan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"krvvpr" "consonants_krvvpr.png" 20;
"sovo" "symbols_sovo.png" 30;
"poru" "word_poru.png" 10;
"lakea" "word_lakea.png" 10;
"säiky" "word_säiky.png" 10;
"kyhmy" "word_kyhmy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ h _" "word_kyhmy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gongi" "word_gongi.png" 10;
"zztr" "consonants_zztr.png" 20;
"vipu" "word_vipu.png" 10;
"qwjvhz" "consonants_qwjvhz.png" 20;
"do^d" "symbols_do^d.png" 30;
"torium" "word_torium.png" 10;
"sävy" "word_sävy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ e _" "word_sävy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rjfcdx" "consonants_rjfcdx.png" 20;
"täällä" "word_täällä.png" 10;
"vdod^" "symbols_vdod^.png" 30;
"v^vsov" "symbols_v^vsov.png" 30;
"ositus" "word_ositus.png" 10;
"nuha" "word_nuha.png" 10;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kuje" "word_kuje.png" 10;
"rwfcq" "consonants_rwfcq.png" 20;
"vvvd" "symbols_vvvd.png" 30;
"gxqtx" "consonants_gxqtx.png" 20;
"kolvi" "word_kolvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ü _ _ _" "word_kolvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oo^vdv" "symbols_oo^vdv.png" 30;
"^vod" "symbols_^vod.png" 30;
"dlgt" "consonants_dlgt.png" 20;
"svvssv" "symbols_svvssv.png" 30;
"ovvds" "symbols_ovvds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_ovvds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuhi" "word_uuhi.png" 10;
"afasia" "word_afasia.png" 10;
"stoola" "word_stoola.png" 10;
"zfjxqk" "consonants_zfjxqk.png" 20;
"mwgz" "consonants_mwgz.png" 20;
"nurja" "word_nurja.png" 10;
"tyrä" "word_tyrä.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^d^vs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^dv^" "symbols_s^dv^.png" 30;
"jdfjs" "consonants_jdfjs.png" 20;
"sdvdvd" "symbols_sdvdvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "symbols_sdvdvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ssods" "symbols_ssods.png" 30;
"räntä" "word_räntä.png" 10;
"o^ss" "symbols_o^ss.png" 30;
"kyteä" "word_kyteä.png" 10;
"^sodvv" "symbols_^sodvv.png" 30;
"d^^do" "symbols_d^^do.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _ _" "symbols_d^^do_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dv^oo" "symbols_dv^oo.png" 30;
"vzkt" "consonants_vzkt.png" 20;
"katodi" "word_katodi.png" 10;
"vsd^" "symbols_vsd^.png" 30;
"czkdbs" "consonants_czkdbs.png" 20;
"s^vo" "symbols_s^vo.png" 30;
"otsoni" "word_otsoni.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"^so^d" "symbols_^so^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_^so^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sopu" "word_sopu.png" 10;
"odo^" "symbols_odo^.png" 30;
"suippo" "word_suippo.png" 10;
"päkiä" "word_päkiä.png" 10;
"hautua" "word_hautua.png" 10;
"uuma" "word_uuma.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ u _ _" "word_uuma_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bzhkd" "consonants_bzhkd.png" 20;
"almu" "word_almu.png" 10;
"kuskus" "word_kuskus.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
"salvaa" "word_salvaa.png" 10;
"gsdx" "consonants_gsdx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _" "consonants_gsdx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tlzr" "consonants_tlzr.png" 20;
"ylkä" "word_ylkä.png" 10;
"^^od" "symbols_^^od.png" 30;
"korren" "word_korren.png" 10;
"vovdsv" "symbols_vovdsv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ v" "symbols_vovdsv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bhsx" "consonants_bhsx.png" 20;
"vssd" "symbols_vssd.png" 30;
"dqkcl" "consonants_dqkcl.png" 20;
"^ss^" "symbols_^ss^.png" 30;
"dvsv" "symbols_dvsv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _" "symbols_dvsv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zsgj" "consonants_zsgj.png" 20;
"d^oood" "symbols_d^oood.png" 30;
};
